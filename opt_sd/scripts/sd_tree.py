# @title
"""
OPT-Tree style speculative decoding for a simple HuggingFace two-model setup.

This file is intentionally written to match the structure of the user's simple
linear speculative-decoding code, but replaces:
  1. linear draft generation with adaptive OPT-Tree draft construction
  2. linear target verification with tree verification
  3. gamma logging with tree/node-budget logging

Important assumptions:
  - The draft and target models must use the same tokenizer / vocabulary.
  - The fastest path uses a 4D tree attention mask. Some HF model classes or
    attention implementations do not accept arbitrary 4D masks; when that fails,
    the code falls back to one batched path-verification forward pass. The
    fallback is easier to run but repeats the prefix, so it is correct but less
    efficient than true tree attention.
  - For do_sample=False or temperature<=0, verification is greedy.
  - For do_sample=True and temperature>0, the target token is sampled directly
    from the target distribution at each verified tree node. This preserves the
    target model distribution and avoids the linear q/p rejection-correction
    rule, because the accepted path is determined by target-model samples.

Mental model:
  1. Tokenize the existing prefix.
  2. Ask the small model to build a tree of likely continuations. A node stores
     one token plus a pointer to its parent, so every node represents
     prefix + path(root -> node).
  3. Ask the big model for next-token logits at every tree parent.
     The fast version flattens the whole tree into one sequence and uses a 4D
     attention mask so siblings cannot see each other. If the model cannot
     consume that custom mask, the portable fallback verifies every parent path
     as a padded batch.
  4. Walk down the tree using big-model choices. Each time the big model chooses
     a child that exists in the draft tree, accept it. When it chooses a token
     outside the drafted children, append that target token and stop this step.
  5. Repeat until the requested number of new tokens is generated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import colorama
from transformers import AutoTokenizer, AutoModelForCausalLM


colorama.init(autoreset=True)


# -----------------------------
# 1. Model loading
# -----------------------------

def loading_models(
    big_model_name: str,
    small_model_name: str,
    device_type: str = "cuda",
    torch_dtype=torch.bfloat16,
):
    tokenizer = AutoTokenizer.from_pretrained(big_model_name)
    print("Loaded tokenizer of big model")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Adjusted tokenizer pad token")

    big_model = AutoModelForCausalLM.from_pretrained(
        big_model_name,
        torch_dtype=torch_dtype,
    ).to(device_type)

    big_model.eval()
    print("Loaded big model")

    small_model = AutoModelForCausalLM.from_pretrained(
        small_model_name,
        torch_dtype=torch_dtype,
    ).to(device_type)

    small_model.eval()
    print("Loaded small model")

    small_model.generation_config.pad_token_id = tokenizer.pad_token_id
    big_model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("Adjusted model pad tokens")
    print("NOTE: OPT-Tree needs draft and target models with the same tokenizer/vocab.")

    return tokenizer, big_model, small_model


# -----------------------------
# 2. Input processing
# -----------------------------

def processing_inputs(
    tokenizer,
    input_text: str,
    device_type: str = "cuda",
):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    ).to(device_type)

    input_ids = inputs["input_ids"][0]
    # The main decoder keeps a 1D token vector and rebuilds masks inside each
    # model call. Returning the original tokenizer mask is useful for debugging
    # and backward compatibility, but the current decode loop does not consume it.
    attention_mask = inputs["attention_mask"]

    return input_ids, attention_mask


# -----------------------------
# 3. Data containers
# -----------------------------

@dataclass
class DraftTreeNode:
    # A node is not a whole sequence. It is one proposed token plus enough
    # metadata to reconstruct the sequence by following parent_id links.
    node_id: int
    token_id: int
    parent_id: int       # -1 means root/prefix
    depth: int           # first token after prefix has depth 1
    draft_prob: float    # local q(token | parent path)
    path_score: float    # product of local draft probabilities along path


@dataclass
class OPTTree:
    # The tree is stored as a flat node list because it is easier to serialize
    # into logs and later visualize. parent_id encodes the edges.
    nodes: List[DraftTreeNode]
    node_budget: int
    max_depth: int
    threshold: float
    expected_acceptance_score: float
    draft_forward_passes: int
    depth_summaries: List[dict]
    draft_cache_used: bool = False

    @property
    def depth(self) -> int:
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes)


@dataclass
class SpeculativeDecodingStats:
    steps: int = 0
    total_tree_nodes: int = 0
    total_tree_depth: int = 0
    total_draft_forward_passes: int = 0
    total_target_forward_passes: int = 0
    total_accepted_draft_tokens: int = 0
    total_generated_tokens: int = 0
    total_wall_time_sec: float = 0.0

    def update(
        self,
        tree_nodes: int,
        tree_depth: int,
        draft_forward_passes: int,
        target_forward_passes: int,
        accepted_draft_tokens: int,
        generated_tokens: int,
        wall_time_sec: float,
    ):
        self.steps += 1
        self.total_tree_nodes += tree_nodes
        self.total_tree_depth += tree_depth
        self.total_draft_forward_passes += draft_forward_passes
        self.total_target_forward_passes += target_forward_passes
        self.total_accepted_draft_tokens += accepted_draft_tokens
        self.total_generated_tokens += generated_tokens
        self.total_wall_time_sec += wall_time_sec

    def to_dict(self):
        d = asdict(self)
        if self.steps > 0:
            d["mean_tree_nodes"] = self.total_tree_nodes / self.steps
            d["mean_tree_depth"] = self.total_tree_depth / self.steps
            d["mean_acceptance_length"] = self.total_generated_tokens / self.steps
            d["mean_accepted_draft_tokens"] = self.total_accepted_draft_tokens / self.steps
            d["mean_draft_forward_passes"] = self.total_draft_forward_passes / self.steps
            d["mean_target_forward_passes"] = self.total_target_forward_passes / self.steps
            d["tokens_per_second_wall"] = (
                self.total_generated_tokens / self.total_wall_time_sec
                if self.total_wall_time_sec > 0
                else None
            )
        else:
            d["mean_tree_nodes"] = 0
            d["mean_tree_depth"] = 0
            d["mean_acceptance_length"] = 0
            d["mean_accepted_draft_tokens"] = 0
            d["mean_draft_forward_passes"] = 0
            d["mean_target_forward_passes"] = 0
            d["tokens_per_second_wall"] = None
        return d


# -----------------------------
# 4. Small helpers
# -----------------------------

def _model_dtype(model) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is not None and temperature > 0:
        logits = logits / temperature
    return torch.softmax(logits.float(), dim=-1)


def _select_from_target_logits(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
) -> Tuple[torch.Tensor, float]:
    """Return one token id and its target probability under the selected policy."""
    if (not do_sample) or temperature is None or temperature <= 0:
        token = torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.softmax(logits.float(), dim=-1)
        prob = probs[token.item()].item()
        return token.to(logits.device), prob

    probs = _softmax_with_temperature(logits, temperature=temperature)
    token = torch.multinomial(probs, num_samples=1)
    prob = probs[token.item()].item()
    return token.to(logits.device), prob


def _get_node_path_token_ids(nodes_by_id: Dict[int, DraftTreeNode], node_id: int) -> List[int]:
    """Tokens from root child to node_id."""
    if node_id == -1:
        return []

    path = []
    current = node_id
    while current != -1:
        node = nodes_by_id[current]
        path.append(node.token_id)
        current = node.parent_id

    path.reverse()
    return path


def _make_padded_batch(
    tokenizer,
    sequences: List[torch.Tensor],
    device_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad sequences and return input_ids, attention_mask, last_indices."""
    max_len = max(seq.numel() for seq in sequences)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full(
        (len(sequences), max_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=device_type,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_len),
        dtype=torch.long,
        device=device_type,
    )

    last_indices = []
    for row, seq in enumerate(sequences):
        seq = seq.to(device_type)
        input_ids[row, : seq.numel()] = seq
        attention_mask[row, : seq.numel()] = 1
        last_indices.append(seq.numel() - 1)

    last_indices = torch.tensor(last_indices, dtype=torch.long, device=device_type)
    return input_ids, attention_mask, last_indices


# -----------------------------
# 5. Draft model: adaptive OPT-Tree construction
# -----------------------------

@torch.no_grad()
def _small_model_next_probs_for_parents(
    small_model,
    tokenizer,
    all_ids: torch.Tensor,
    nodes_by_id: Dict[int, DraftTreeNode],
    parent_ids: List[int],
    device_type: str,
    draft_temperature: float,
) -> torch.Tensor:
    """
    Compute q(next_token | prefix + path(parent)) for many parent nodes.
    This is a simple batched implementation; it is not the KV-cache-optimized
    version from the original repo.
    """
    # Every parent node represents a different prefix+path. Batch those paths
    # together so one small-model forward expands many frontier nodes at once.
    sequences = []
    for parent_id in parent_ids:
        path_tokens = _get_node_path_token_ids(nodes_by_id, parent_id)
        if path_tokens:
            path_tensor = torch.tensor(path_tokens, dtype=torch.long, device=device_type)
            seq = torch.cat([all_ids.to(device_type), path_tensor])
        else:
            seq = all_ids.to(device_type)
        sequences.append(seq)

    input_ids, attention_mask, last_indices = _make_padded_batch(
        tokenizer=tokenizer,
        sequences=sequences,
        device_type=device_type,
    )

    out = small_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[torch.arange(len(parent_ids), device=device_type), last_indices]
    probs = _softmax_with_temperature(logits, temperature=draft_temperature)
    return probs


@torch.no_grad()
def _small_model_prefill_for_cached_tree(
    small_model,
    all_ids: torch.Tensor,
    device_type: str,
):
    """Run the draft model once on the prefix and keep logits + KV cache."""
    input_ids = all_ids.to(device_type).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device_type)

    out = small_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def _small_model_extend_cached_node(
    small_model,
    token_id: int,
    parent_past_key_values,
    parent_sequence_length: int,
    device_type: str,
):
    """
    Extend one cached prefix+path by one token and return logits + new cache.

    This avoids rerunning the whole prefix/path for every new draft node. It is
    still Python-call-heavy, but each call processes only one token.
    """
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device_type)
    attention_mask = torch.ones(
        (1, parent_sequence_length + 1),
        dtype=torch.long,
        device=device_type,
    )

    out = small_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=parent_past_key_values,
        use_cache=True,
        return_dict=True,
    )

    return out.logits[:, -1, :], out.past_key_values


def _select_closed_top_nodes(
    all_nodes: List[DraftTreeNode],
    candidate_ids: List[int],
    node_budget: int,
) -> List[int]:
    """
    Select high-score nodes while keeping ancestor closure.
    OPT-Tree's monotonic path scores normally make the top-k nodes closed.
    This function keeps the code robust under ties/numerical edge cases.
    """
    nodes_by_id = {node.node_id: node for node in all_nodes}
    ordered = sorted(
        set(candidate_ids),
        key=lambda idx: all_nodes[idx].path_score,
        reverse=True,
    )

    # "Closed" means: if we keep a node, we must also keep every ancestor.
    # Without this, verification could point to a child whose parent was pruned.
    selected = set()

    for idx in ordered:
        chain = []
        current = idx
        while current != -1 and current not in selected:
            chain.append(current)
            current = nodes_by_id[current].parent_id

        if len(selected) + len(chain) <= node_budget:
            selected.update(chain)

        if len(selected) >= node_budget:
            break

    return sorted(selected, key=lambda idx: (all_nodes[idx].depth, idx))


@torch.no_grad()
def construct_opt_tree(
    small_model,
    tokenizer,
    all_ids: torch.Tensor,
    node_budget: int = 32,
    max_depth: int = 8,
    threshold: float = 0.0,
    branch_top_k: Optional[int] = None,
    draft_temperature: float = 1.0,
    device_type: str = "cuda",
    use_draft_cache: bool = True,
) -> OPTTree:
    """
    Build an adaptive draft tree.

    The approximation used here follows OPT-Tree's practical intuition:
    path_score(node) = product of draft probabilities along the node path.
    At each layer, expand high-score frontier nodes and keep the closed subtree
    with the largest path scores under a fixed node budget.
    """
    assert node_budget >= 1, "node_budget must be >= 1"
    assert max_depth >= 1, "max_depth must be >= 1"

    if branch_top_k is None:
        branch_top_k = node_budget
    branch_top_k = max(1, min(branch_top_k, node_budget))

    all_nodes: List[DraftTreeNode] = []
    nodes_by_id: Dict[int, DraftTreeNode] = {}
    all_candidate_ids: List[int] = []
    frontier_parent_ids: List[int] = [-1]
    next_node_id = 0
    previous_best_score = 0.0
    draft_forward_passes = 0
    depth_summaries: List[dict] = []

    final_selected_ids: List[int] = []
    prefix_len = int(all_ids.numel())

    root_logits = None
    root_past_key_values = None
    cached_logits_by_id: Dict[int, torch.Tensor] = {}
    cached_past_by_id: Dict[int, object] = {}
    effective_use_draft_cache = use_draft_cache

    if effective_use_draft_cache:
        try:
            root_logits, root_past_key_values = _small_model_prefill_for_cached_tree(
                small_model=small_model,
                all_ids=all_ids,
                device_type=device_type,
            )
            draft_forward_passes += 1
        except Exception:
            # Some model classes/backends do not expose reusable caches in a way
            # this simple tree builder can consume. Fall back to the original
            # batched full-path implementation rather than failing the decode.
            effective_use_draft_cache = False

    for depth in range(1, max_depth + 1):
        if not frontier_parent_ids:
            break

        if effective_use_draft_cache:
            # Expand the frontier using logits that were produced when each
            # parent node's cache was materialized. No prefix/path recompute.
            logits_rows = []
            for parent_id in frontier_parent_ids:
                if parent_id == -1:
                    logits_rows.append(root_logits)
                else:
                    logits_rows.append(cached_logits_by_id[parent_id])

            logits_batch = torch.cat(logits_rows, dim=0)
            probs_batch = _softmax_with_temperature(
                logits_batch,
                temperature=draft_temperature,
            )
        else:
            # Original portable path: run full prefix+path sequences as a batch.
            probs_batch = _small_model_next_probs_for_parents(
                small_model=small_model,
                tokenizer=tokenizer,
                all_ids=all_ids,
                nodes_by_id=nodes_by_id,
                parent_ids=frontier_parent_ids,
                device_type=device_type,
                draft_temperature=draft_temperature,
            )
            draft_forward_passes += 1

        new_ids_this_depth: List[int] = []

        for row, parent_id in enumerate(frontier_parent_ids):
            parent_score = 1.0 if parent_id == -1 else nodes_by_id[parent_id].path_score
            k = min(branch_top_k, probs_batch.shape[-1])
            top_probs, top_token_ids = torch.topk(probs_batch[row], k=k, dim=-1)

            for local_prob, token_id in zip(top_probs.tolist(), top_token_ids.tolist()):
                path_score = float(parent_score * local_prob)
                node = DraftTreeNode(
                    node_id=next_node_id,
                    token_id=int(token_id),
                    parent_id=int(parent_id),
                    depth=depth,
                    draft_prob=float(local_prob),
                    path_score=path_score,
                )
                all_nodes.append(node)
                nodes_by_id[next_node_id] = node
                all_candidate_ids.append(next_node_id)
                new_ids_this_depth.append(next_node_id)
                next_node_id += 1

        # Keep only the best closed subtree under the node budget. This is the
        # adaptive part: the tree can be wide, deep, or mixed depending on where
        # the small model puts probability mass.
        selected_ids = _select_closed_top_nodes(
            all_nodes=all_nodes,
            candidate_ids=all_candidate_ids,
            node_budget=node_budget,
        )
        selected_score = sum(all_nodes[idx].path_score for idx in selected_ids)
        score_gain = selected_score - previous_best_score

        depth_summaries.append(
            {
                "depth": depth,
                "expanded_parent_nodes": len(frontier_parent_ids),
                "new_nodes": len(new_ids_this_depth),
                "selected_nodes_under_budget": len(selected_ids),
                "selected_expected_acceptance_score": selected_score,
                "score_gain_from_previous_depth": score_gain,
            }
        )

        final_selected_ids = selected_ids

        # OPT-Tree uses a threshold to avoid paying another draft depth when
        # the expected-acceptance gain is too small.
        if depth > 1 and score_gain <= threshold:
            break

        previous_best_score = selected_score

        # Expand only high-score nodes from the newest layer, matching the idea
        # that the last layer is the frontier.
        next_frontier_parent_ids = sorted(
            new_ids_this_depth,
            key=lambda idx: all_nodes[idx].path_score,
            reverse=True,
        )[:node_budget]

        if effective_use_draft_cache:
            next_cached_logits_by_id: Dict[int, torch.Tensor] = {}
            next_cached_past_by_id: Dict[int, object] = {}

            for node_id in next_frontier_parent_ids:
                node = nodes_by_id[node_id]
                if node.parent_id == -1:
                    parent_past = root_past_key_values
                    parent_sequence_length = prefix_len
                else:
                    parent_past = cached_past_by_id[node.parent_id]
                    parent_sequence_length = prefix_len + nodes_by_id[node.parent_id].depth

                node_logits, node_past = _small_model_extend_cached_node(
                    small_model=small_model,
                    token_id=node.token_id,
                    parent_past_key_values=parent_past,
                    parent_sequence_length=parent_sequence_length,
                    device_type=device_type,
                )
                draft_forward_passes += 1
                next_cached_logits_by_id[node_id] = node_logits
                next_cached_past_by_id[node_id] = node_past

            cached_logits_by_id = next_cached_logits_by_id
            cached_past_by_id = next_cached_past_by_id

        frontier_parent_ids = next_frontier_parent_ids

    # Re-map selected nodes to dense node ids in topological order. The
    # visualizer and 4D attention builder assume parent ids point backward.
    final_old_ids = sorted(final_selected_ids, key=lambda idx: (all_nodes[idx].depth, idx))
    old_to_new: Dict[int, int] = {}
    final_nodes: List[DraftTreeNode] = []

    for new_id, old_id in enumerate(final_old_ids):
        old = all_nodes[old_id]
        parent_new_id = -1 if old.parent_id == -1 else old_to_new[old.parent_id]
        old_to_new[old_id] = new_id
        final_nodes.append(
            DraftTreeNode(
                node_id=new_id,
                token_id=old.token_id,
                parent_id=parent_new_id,
                depth=old.depth,
                draft_prob=old.draft_prob,
                path_score=old.path_score,
            )
        )

    expected_acceptance_score = sum(node.path_score for node in final_nodes)

    return OPTTree(
        nodes=final_nodes,
        node_budget=node_budget,
        max_depth=max_depth,
        threshold=threshold,
        expected_acceptance_score=expected_acceptance_score,
        draft_forward_passes=draft_forward_passes,
        depth_summaries=depth_summaries,
        draft_cache_used=effective_use_draft_cache,
    )


# -----------------------------
# 6. Big model: tree verification forward pass
# -----------------------------

def _build_children_by_parent(tree: OPTTree) -> Dict[int, Dict[int, DraftTreeNode]]:
    children: Dict[int, Dict[int, DraftTreeNode]] = {}
    for node in tree.nodes:
        children.setdefault(node.parent_id, {})
        # If the same token appears twice under the same parent, keep the higher
        # path-score node. This should be rare with top-k from one distribution.
        prev = children[node.parent_id].get(node.token_id)
        if prev is None or node.path_score > prev.path_score:
            children[node.parent_id][node.token_id] = node
    return children


def _build_tree_attention_inputs(
    big_model,
    all_ids: torch.Tensor,
    tree: OPTTree,
    device_type: str,
):
    """Create flat tree input ids, 4D tree attention mask, and position ids.

    The flattened input is:

        [prefix tokens..., tree node 0, tree node 1, ...]

    The custom mask gives each tree node visibility into the prefix and its own
    ancestors only. That is the key trick: it lets one target-model forward
    compute logits for many incompatible branch paths without sibling leakage.
    """
    prefix_len = all_ids.numel()
    num_tree_nodes = len(tree.nodes)
    total_len = prefix_len + num_tree_nodes

    tree_token_ids = torch.tensor(
        [node.token_id for node in tree.nodes],
        dtype=torch.long,
        device=device_type,
    )
    input_ids = torch.cat([all_ids.to(device_type), tree_token_ids], dim=0).unsqueeze(0)

    position_ids = torch.empty((total_len,), dtype=torch.long, device=device_type)
    position_ids[:prefix_len] = torch.arange(prefix_len, dtype=torch.long, device=device_type)
    for i, node in enumerate(tree.nodes):
        # Child of the prefix has absolute position prefix_len.
        # Siblings at the same depth share the same position id.
        position_ids[prefix_len + i] = (prefix_len - 1) + node.depth
    position_ids = position_ids.unsqueeze(0)

    node_to_flat_pos = {node.node_id: prefix_len + i for i, node in enumerate(tree.nodes)}
    nodes_by_id = {node.node_id: node for node in tree.nodes}

    # allowed[row, col] answers: "May token at row attend to token at col?"
    # Prefix rows use normal causal attention; tree rows use tree-path attention.
    allowed = torch.zeros((total_len, total_len), dtype=torch.bool, device=device_type)

    # Normal causal attention inside the prefix.
    for i in range(prefix_len):
        allowed[i, : i + 1] = True

    # Tree nodes can attend to the whole prefix and their own ancestor path.
    for node in tree.nodes:
        row = node_to_flat_pos[node.node_id]
        allowed[row, :prefix_len] = True

        current = node.node_id
        while current != -1:
            allowed[row, node_to_flat_pos[current]] = True
            current = nodes_by_id[current].parent_id

    # Use float32 for safety. Most HF attention code accepts float additive masks.
    min_value = torch.finfo(torch.float32).min
    additive_mask = torch.zeros((1, 1, total_len, total_len), dtype=torch.float32, device=device_type)
    additive_mask = additive_mask.masked_fill(~allowed.view(1, 1, total_len, total_len), min_value)

    return input_ids, additive_mask, position_ids, node_to_flat_pos


@torch.no_grad()
def _big_model_tree_attention_forward(
    big_model,
    all_ids: torch.Tensor,
    tree: OPTTree,
    device_type: str,
) -> Tuple[Dict[int, torch.Tensor], dict]:
    """Fast path: one flat tree forward with a 4D tree attention mask."""
    input_ids, attention_mask, position_ids, node_to_flat_pos = _build_tree_attention_inputs(
        big_model=big_model,
        all_ids=all_ids,
        tree=tree,
        device_type=device_type,
    )

    out = big_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    logits = out.logits[0]
    prefix_len = all_ids.numel()

    logits_by_parent: Dict[int, torch.Tensor] = {-1: logits[prefix_len - 1]}
    for node in tree.nodes:
        logits_by_parent[node.node_id] = logits[node_to_flat_pos[node.node_id]]

    info = {
        "target_forward_method": "tree_attention_4d",
        "target_forward_passes": 1,
        "flat_tree_sequence_length": int(input_ids.shape[-1]),
        "tree_attention_supported": True,
        "tree_attention_error": None,
    }
    return logits_by_parent, info


@torch.no_grad()
def _big_model_batched_path_forward(
    big_model,
    tokenizer,
    all_ids: torch.Tensor,
    tree: OPTTree,
    device_type: str,
    tree_attention_error: Optional[str] = None,
) -> Tuple[Dict[int, torch.Tensor], dict]:
    """
    Portable fallback: one batched target forward over prefix+path(parent) for
    root and every draft node. Correct but less efficient than 4D tree attention.
    """
    # Fallback shape: one row for the root/prefix and one row for each node as a
    # parent. This repeats the prefix many times, so it is memory/work heavier
    # than 4D attention, but it uses standard 2D attention masks.
    nodes_by_id = {node.node_id: node for node in tree.nodes}
    parent_ids = [-1] + [node.node_id for node in tree.nodes]

    sequences = []
    for parent_id in parent_ids:
        path_tokens = _get_node_path_token_ids(nodes_by_id, parent_id)
        if path_tokens:
            path_tensor = torch.tensor(path_tokens, dtype=torch.long, device=device_type)
            seq = torch.cat([all_ids.to(device_type), path_tensor])
        else:
            seq = all_ids.to(device_type)
        sequences.append(seq)

    input_ids, attention_mask, last_indices = _make_padded_batch(
        tokenizer=tokenizer,
        sequences=sequences,
        device_type=device_type,
    )

    out = big_model(input_ids=input_ids, attention_mask=attention_mask)
    last_logits = out.logits[torch.arange(len(parent_ids), device=device_type), last_indices]

    logits_by_parent = {parent_id: last_logits[row] for row, parent_id in enumerate(parent_ids)}

    info = {
        "target_forward_method": "batched_paths_fallback",
        "target_forward_passes": 1,
        "batched_path_batch_size": len(parent_ids),
        "batched_path_max_length": int(input_ids.shape[-1]),
        "tree_attention_supported": False,
        "tree_attention_error": tree_attention_error,
    }
    return logits_by_parent, info


@torch.no_grad()
def forward_target_model_on_tree(
    big_model,
    tokenizer,
    all_ids: torch.Tensor,
    tree: OPTTree,
    device_type: str = "cuda",
    prefer_tree_attention: bool = True,
) -> Tuple[Dict[int, torch.Tensor], dict]:
    if len(tree.nodes) == 0:
        return _big_model_batched_path_forward(
            big_model=big_model,
            tokenizer=tokenizer,
            all_ids=all_ids,
            tree=tree,
            device_type=device_type,
        )

    # Try true tree attention first. HuggingFace support varies by model and
    # attention backend, so failure is expected on some setups and not fatal.
    if prefer_tree_attention:
        try:
            return _big_model_tree_attention_forward(
                big_model=big_model,
                all_ids=all_ids,
                tree=tree,
                device_type=device_type,
            )
        except Exception as exc:  # noqa: BLE001 - keep fallback robust for HF model differences.
            return _big_model_batched_path_forward(
                big_model=big_model,
                tokenizer=tokenizer,
                all_ids=all_ids,
                tree=tree,
                device_type=device_type,
                tree_attention_error=repr(exc),
            )

    return _big_model_batched_path_forward(
        big_model=big_model,
        tokenizer=tokenizer,
        all_ids=all_ids,
        tree=tree,
        device_type=device_type,
    )


# -----------------------------
# 7. Verify the tree and choose output tokens
# -----------------------------

@torch.no_grad()
def verify_tree_and_sample_output(
    tokenizer,
    tree: OPTTree,
    logits_by_parent: Dict[int, torch.Tensor],
    do_sample: bool,
    temperature: float,
    device_type: str = "cuda",
    decode_tokens: bool = True,
) -> dict:
    """
    Walk down the draft tree using target-model choices.

    At parent p, the target model chooses the next token from p_target(. | path(p)).
    If that token exists as a drafted child, accept it and continue. Otherwise,
    append that target token and stop. This is exact for greedy decoding and also
    preserves the target sampling distribution for do_sample=True.
    """
    # Build quick lookup: at each parent, which drafted children are available?
    # Verification then becomes a target-model walk through that lookup table.
    children_by_parent = _build_children_by_parent(tree)
    current_parent = -1

    accepted_node_ids: List[int] = []
    accepted_token_ids: List[int] = []
    target_choices: List[dict] = []

    while True:
        logits = logits_by_parent[current_parent]
        target_token, target_prob = _select_from_target_logits(
            logits=logits,
            do_sample=do_sample,
            temperature=temperature,
        )
        target_token_id = int(target_token.item())

        child = children_by_parent.get(current_parent, {}).get(target_token_id)
        target_choices.append(
            {
                "parent_node_id": current_parent,
                "target_token_id": target_token_id,
                "target_token_text": (
                    tokenizer.decode([target_token_id]) if decode_tokens else None
                ),
                "target_token_probability": target_prob,
                "hit_draft_child": child is not None,
                "child_node_id": None if child is None else child.node_id,
            }
        )

        if child is None:
            # The target chose a token outside the draft tree at this parent.
            # Append the target token as the "bonus" token and finish the step.
            bonus_token = target_token.reshape(1).to(device_type)
            stop_reason = "target_token_not_in_draft_children"
            break

        # The target token matches a drafted child, so this draft token is safe
        # to accept and we continue verification deeper in the tree.
        accepted_node_ids.append(child.node_id)
        accepted_token_ids.append(child.token_id)
        current_parent = child.node_id

        # If this is a leaf, the next loop will sample/generate the bonus token
        # from logits_by_parent[current_parent] and stop because there are no children.

    accepted_tokens = torch.tensor(
        accepted_token_ids,
        dtype=torch.long,
        device=device_type,
    )

    return {
        "accepted_tokens": accepted_tokens,
        "accepted_node_ids": accepted_node_ids,
        "bonus_token": bonus_token,
        "target_choices": target_choices,
        "stop_reason": stop_reason,
    }


# -----------------------------
# 8. Logging
# -----------------------------

def create_tree_step_log(
    tokenizer,
    step_index: int,
    all_ids_before_step: torch.Tensor,
    tree: OPTTree,
    target_forward_info: dict,
    verify_info: dict,
    step_tokens: torch.Tensor,
    all_ids_after_step: torch.Tensor,
    wall_time_sec: float,
    include_tree_nodes: bool = True,
    decode_text: bool = True,
):
    accepted_tokens = verify_info["accepted_tokens"]
    bonus_token = verify_info["bonus_token"]

    if decode_text:
        accepted_text = tokenizer.decode(accepted_tokens, skip_special_tokens=False)
        bonus_text = tokenizer.decode(bonus_token, skip_special_tokens=False)
        step_text = tokenizer.decode(step_tokens, skip_special_tokens=False)
        full_text_after_step = tokenizer.decode(all_ids_after_step, skip_special_tokens=True)
    else:
        accepted_text = None
        bonus_text = None
        step_text = None
        full_text_after_step = None

    return {
        "step_index": step_index,
        "prefix_num_tokens": int(all_ids_before_step.numel()),
        "tree": {
            "node_budget": tree.node_budget,
            "actual_nodes": len(tree.nodes),
            "max_depth_setting": tree.max_depth,
            "actual_depth": tree.depth,
            "threshold": tree.threshold,
            "expected_acceptance_score_sum_path_probs": tree.expected_acceptance_score,
            "draft_forward_passes": tree.draft_forward_passes,
            "draft_cache_used": tree.draft_cache_used,
            "depth_summaries": tree.depth_summaries,
            "nodes": [asdict(node) for node in tree.nodes] if include_tree_nodes else [],
        },
        "target_forward": target_forward_info,
        "verification": {
            "accepted_node_ids": verify_info["accepted_node_ids"],
            "num_accepted_draft_tokens": int(accepted_tokens.numel()),
            "accepted_text": accepted_text,
            "bonus_token_id": int(bonus_token.item()),
            "bonus_text": bonus_text,
            "target_choices": verify_info["target_choices"],
            "stop_reason": verify_info["stop_reason"],
        },
        "output": {
            "step_num_tokens": int(step_tokens.numel()),
            "step_text": step_text,
            "total_num_tokens_after_step": int(all_ids_after_step.numel()),
            "full_text_after_step": full_text_after_step,
        },
        "timing": {
            "wall_time_sec": wall_time_sec,
        },
    }


def print_tree_compact_step_log(step_log: dict):
    tree = step_log["tree"]
    verification = step_log["verification"]
    target_forward = step_log["target_forward"]
    output = step_log["output"]

    print("\n" + "-" * 80)
    print(f"OPT-Tree step {step_log['step_index']}")
    print(
        f"nodes={tree['actual_nodes']}/{tree['node_budget']} | "
        f"depth={tree['actual_depth']}/{tree['max_depth_setting']} | "
        f"E_score={tree['expected_acceptance_score_sum_path_probs']:.4f} | "
        f"draft_forwards={tree['draft_forward_passes']} | "
        f"target_forward={target_forward['target_forward_method']}"
    )
    print(
        f"accepted_draft_tokens={verification['num_accepted_draft_tokens']} | "
        f"bonus={verification['bonus_text']!r} | "
        f"step_text={output['step_text']!r}"
    )

    if not target_forward.get("tree_attention_supported", True):
        print("Tree attention fallback used. Reason:")
        print(target_forward.get("tree_attention_error"))


def print_tree_speculative_step(
    tokenizer,
    all_ids_before_step: torch.Tensor,
    accepted_tokens: torch.Tensor,
    bonus_token: torch.Tensor,
):
    print(colorama.Fore.BLACK + tokenizer.decode(all_ids_before_step), end="")
    print(colorama.Fore.BLUE + tokenizer.decode(accepted_tokens), end="")
    print(colorama.Fore.GREEN + tokenizer.decode(bonus_token), end="")
    print()


# -----------------------------
# 9. One OPT-Tree speculative decoding step
# -----------------------------

@torch.no_grad()
def opt_tree_speculative_decoding_step(
    tokenizer,
    small_model,
    big_model,
    all_ids: torch.Tensor,
    step_index: int,
    node_budget: int = 32,
    max_depth: int = 8,
    threshold: float = 0.0,
    branch_top_k: Optional[int] = None,
    device_type: str = "cuda",
    do_sample: bool = False,
    temperature: float = 0.0,
    draft_temperature: float = 1.0,
    prefer_tree_attention: bool = True,
    use_draft_cache: bool = True,
    remaining_tokens: Optional[int] = None,
    print_logs: bool = True,
    log_tree_nodes: bool = True,
    decode_step_text: bool = True,
):
    step_start = time.perf_counter()
    all_ids_before_step = all_ids.clone()

    # Phase 1: draft. The small model proposes a compact tree of possible next
    # tokens under the node budget.
    tree = construct_opt_tree(
        small_model=small_model,
        tokenizer=tokenizer,
        all_ids=all_ids,
        node_budget=node_budget,
        max_depth=max_depth,
        threshold=threshold,
        branch_top_k=branch_top_k,
        draft_temperature=draft_temperature,
        device_type=device_type,
        use_draft_cache=use_draft_cache,
    )

    # Phase 2: verify. The big model produces next-token logits for the root and
    # for every node as a potential parent.
    logits_by_parent, target_forward_info = forward_target_model_on_tree(
        big_model=big_model,
        tokenizer=tokenizer,
        all_ids=all_ids,
        tree=tree,
        device_type=device_type,
        prefer_tree_attention=prefer_tree_attention,
    )

    # Phase 3: choose output. Walk down the tree according to big-model choices,
    # accepting drafted tokens until the big model leaves the tree.
    verify_info = verify_tree_and_sample_output(
        tokenizer=tokenizer,
        tree=tree,
        logits_by_parent=logits_by_parent,
        do_sample=do_sample,
        temperature=temperature,
        device_type=device_type,
        decode_tokens=decode_step_text or print_logs,
    )

    accepted_tokens = verify_info["accepted_tokens"]
    bonus_token = verify_info["bonus_token"]
    step_tokens = torch.cat([accepted_tokens, bonus_token], dim=0)

    if remaining_tokens is not None:
        step_tokens = step_tokens[:remaining_tokens]

    new_all_ids = torch.cat([all_ids, step_tokens], dim=0)
    wall_time_sec = time.perf_counter() - step_start

    step_log = create_tree_step_log(
        tokenizer=tokenizer,
        step_index=step_index,
        all_ids_before_step=all_ids_before_step,
        tree=tree,
        target_forward_info=target_forward_info,
        verify_info=verify_info,
        step_tokens=step_tokens,
        all_ids_after_step=new_all_ids,
        wall_time_sec=wall_time_sec,
        include_tree_nodes=log_tree_nodes,
        decode_text=decode_step_text or print_logs,
    )

    if print_logs:
        print_tree_compact_step_log(step_log)
        print_tree_speculative_step(
            tokenizer=tokenizer,
            all_ids_before_step=all_ids_before_step,
            accepted_tokens=accepted_tokens,
            bonus_token=bonus_token,
        )

    step_info = {
        "tree": tree,
        "target_forward_info": target_forward_info,
        "verify_info": verify_info,
        "accepted_tokens": accepted_tokens,
        "bonus_token": bonus_token,
        "step_tokens": step_tokens,
        "step_log": step_log,
    }

    return new_all_ids, step_info


# -----------------------------
# 10. Full main pipeline
# -----------------------------

@torch.no_grad()
def speculative_decode_tree(
    input_text: str,
    tokenizer,
    small_model,
    big_model,
    seqlen: int = 50,
    gamma: int = 5,  # kept only for API compatibility with your old function
    device_type: str = "cuda",
    do_sample: bool = False,
    temperature: float = 0.0,
    print_logs: bool = True,
    save_logs_path: Optional[str] = None,

    # OPT-Tree settings
    node_budget: int = 32,
    max_depth: int = 8,
    threshold: float = 0.0,
    branch_top_k: Optional[int] = None,
    draft_temperature: float = 1.0,
    prefer_tree_attention: bool = True,
    use_draft_cache: bool = True,
    stop_on_eos: bool = True,
    log_tree_nodes: bool = True,
    decode_step_text: bool = True,
):
    """
    Drop-in replacement for your previous speculative_decode.

    gamma is intentionally kept in the signature so old calls do not break, but
    OPT-Tree uses node_budget/max_depth/threshold instead of a linear gamma.
    """
    input_ids, _attention_mask = processing_inputs(
        tokenizer=tokenizer,
        input_text=input_text,
        device_type=device_type,
    )

    all_ids = input_ids.clone()
    all_step_infos = []
    all_step_logs = []
    decoding_stats = SpeculativeDecodingStats()

    step_index = 0

    # Each loop iteration may add multiple tokens: accepted draft tokens plus
    # one target-model bonus token. That is where speculative speedup can appear.
    while len(all_ids) - len(input_ids) < seqlen:
        remaining_tokens = seqlen - (len(all_ids) - len(input_ids))

        all_ids, step_info = opt_tree_speculative_decoding_step(
            tokenizer=tokenizer,
            small_model=small_model,
            big_model=big_model,
            all_ids=all_ids,
            step_index=step_index,
            node_budget=node_budget,
            max_depth=max_depth,
            threshold=threshold,
            branch_top_k=branch_top_k,
            device_type=device_type,
            do_sample=do_sample,
            temperature=temperature,
            draft_temperature=draft_temperature,
            prefer_tree_attention=prefer_tree_attention,
            use_draft_cache=use_draft_cache,
            remaining_tokens=remaining_tokens,
            print_logs=print_logs,
            log_tree_nodes=log_tree_nodes,
            decode_step_text=decode_step_text,
        )

        step_log = step_info["step_log"]
        tree: OPTTree = step_info["tree"]
        verify_info = step_info["verify_info"]
        target_forward_info = step_info["target_forward_info"]

        decoding_stats.update(
            tree_nodes=len(tree.nodes),
            tree_depth=tree.depth,
            draft_forward_passes=tree.draft_forward_passes,
            target_forward_passes=target_forward_info.get("target_forward_passes", 1),
            accepted_draft_tokens=len(verify_info["accepted_tokens"]),
            generated_tokens=len(step_info["step_tokens"]),
            wall_time_sec=step_log["timing"]["wall_time_sec"],
        )

        step_log["running_stats"] = decoding_stats.to_dict()
        all_step_infos.append(step_info)
        all_step_logs.append(step_log)

        if print_logs:
            print("\nRunning OPT-Tree stats:")
            print(json.dumps(decoding_stats.to_dict(), indent=2))

        if stop_on_eos and tokenizer.eos_token_id is not None:
            if int(all_ids[-1].item()) == int(tokenizer.eos_token_id):
                break

        step_index += 1

    final_text = tokenizer.decode(all_ids, skip_special_tokens=True)

    final_report = {
        "input_text": input_text,
        "final_text": final_text,
        "seqlen": seqlen,
        "old_gamma_argument_ignored": gamma,
        "decode_method": "opt_tree_speculative_decoding",
        "node_budget": node_budget,
        "max_depth": max_depth,
        "threshold": threshold,
        "branch_top_k": branch_top_k,
        "do_sample": do_sample,
        "temperature": temperature,
        "draft_temperature": draft_temperature,
        "prefer_tree_attention": prefer_tree_attention,
        "use_draft_cache": use_draft_cache,
        "log_tree_nodes": log_tree_nodes,
        "decode_step_text": decode_step_text,
        "stats": decoding_stats.to_dict(),
        "steps": all_step_logs,
    }

    if save_logs_path is not None:
        with open(save_logs_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved logs to: {save_logs_path}")

    return final_text, all_ids, all_step_infos, final_report



