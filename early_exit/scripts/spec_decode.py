from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .draft import generate_with_tuned_lens


@dataclass
class DecodeStats:
    accepted: int = 0
    generated: int = 0

    @property
    def alpha(self) -> float:
        return self.accepted / self.generated if self.generated else 0.0


@torch.no_grad()
def speculative_decode_once(
    model: torch.nn.Module,
    tuned_lens: torch.nn.Module,
    all_ids: torch.Tensor,
    layer_index: int,
    gamma: int,
    stats: DecodeStats | None = None,
) -> tuple[torch.Tensor, DecodeStats]:
    """Run one speculative decoding step and return updated token ids."""
    if stats is None:
        stats = DecodeStats()

    device = all_ids.device

    small_out = generate_with_tuned_lens(
        model=model,
        tuned_lens=tuned_lens,
        layer_index=layer_index,
        prefix_tokens=all_ids,
        gamma=gamma,
    )

    draft_seq = small_out["sequences"]
    draft_tokens = draft_seq[0, -gamma:]

    q = torch.stack(small_out["scores"], dim=1)[0].softmax(dim=-1)  # [gamma, vocab]
    q_of_generated = q[torch.arange(gamma, device=device), draft_tokens]

    attention_mask = torch.ones_like(draft_seq, device=device)
    p_logits = model(
        draft_seq,
        attention_mask=attention_mask,
        return_dict=True,
    ).logits

    # Distribution at positions that predict the gamma generated draft tokens.
    p = p_logits[:, -gamma - 1 : -1, :].softmax(dim=-1)[0]  # [gamma, vocab]
    p_of_generated = p[torch.arange(gamma, device=device), draft_tokens]

    ratio = torch.clamp(p_of_generated / torch.clamp(q_of_generated, min=1e-12), max=1.0)
    is_accepted = torch.rand(gamma, device=device) < ratio

    index_to_reject = torch.argmin(
        torch.cat([is_accepted, torch.tensor([False], device=device)]).to(torch.int)
    ).item()

    accepted_tokens = draft_tokens[:index_to_reject]

    if index_to_reject == gamma:
        p_for_sample = p[-1]
    else:
        p_for_sample = p[index_to_reject] - q[index_to_reject]
        p_for_sample = torch.clamp(p_for_sample, min=0)
        denom = p_for_sample.sum()
        if denom <= 0:
            p_for_sample = p[index_to_reject]
        else:
            p_for_sample = p_for_sample / denom

    sampled_token = torch.multinomial(p_for_sample, num_samples=1)

    stats.accepted += index_to_reject
    stats.generated += index_to_reject
    if index_to_reject < gamma:
        stats.generated += 1

    updated_ids = torch.cat([all_ids, accepted_tokens, sampled_token])
    return updated_ids, stats


@torch.no_grad()
def speculative_generate(
    model: torch.nn.Module,
    tuned_lens: torch.nn.Module,
    input_ids: torch.Tensor,
    seqlen: int,
    layer_index: int,
    gamma: int,
) -> Dict[str, torch.Tensor | DecodeStats]:
    """Generate seqlen new tokens with early-exit TunedLens speculative decoding."""
    if input_ids.dim() != 1:
        raise ValueError(f"input_ids must be 1D [seq], got shape {tuple(input_ids.shape)}")

    all_ids = input_ids
    stats = DecodeStats()

    while len(all_ids) - len(input_ids) < seqlen:
        all_ids, stats = speculative_decode_once(
            model=model,
            tuned_lens=tuned_lens,
            all_ids=all_ids,
            layer_index=layer_index,
            gamma=gamma,
            stats=stats,
        )

    # Match requested length exactly if the last accepted chunk overshot.
    all_ids = all_ids[: len(input_ids) + seqlen]

    return {
        "sequences": all_ids[None],
        "stats": stats,
    }
