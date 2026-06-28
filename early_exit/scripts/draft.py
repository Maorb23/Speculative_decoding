from __future__ import annotations

from typing import Dict, List

import torch

from .modeling import get_transformer_layers


@torch.no_grad()
def generate_with_tuned_lens(
    model: torch.nn.Module,
    tuned_lens: torch.nn.Module,
    layer_index: int,
    prefix_tokens: torch.Tensor,
    gamma: int,
) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    """
    Draft gamma tokens from an early layer using TunedLens.

    This preserves the notebook's idea: temporarily truncate the model to layers
    0..layer_index, use hidden_states[layer_index + 1], map it through the tuned
    lens, and sample from the resulting next-token distribution.

    Note: this mutates the layer list during the call, then restores it. For
    production benchmarking, building a separate shallow model is cleaner.
    """
    if prefix_tokens.dim() != 1:
        raise ValueError(f"prefix_tokens must be 1D [seq], got shape {tuple(prefix_tokens.shape)}")

    layers = get_transformer_layers(model)
    expected_layers = model.config.num_hidden_layers
    if len(layers) != expected_layers:
        raise ValueError(
            f"Model is already truncated: len(layers)={len(layers)}, "
            f"expected={expected_layers}. Reload the model."
        )

    if not (0 <= layer_index < expected_layers):
        raise ValueError(f"layer_index must be in [0, {expected_layers - 1}], got {layer_index}")

    num_layers_to_keep = layer_index + 1
    num_layers_to_remove = len(layers) - num_layers_to_keep
    removed_layers = [layers.pop(-1) for _ in range(num_layers_to_remove)]

    scores: List[torch.Tensor] = []

    for _ in range(gamma):
        model_input = prefix_tokens[None]
        attention_mask = torch.ones_like(model_input, device=prefix_tokens.device)

        out = model(
            model_input,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states[0] is embeddings; hidden_states[layer_index+1] is after that block.
        h = out.hidden_states[layer_index + 1]
        tuned_lens_logits = tuned_lens(h, layer_index)      # [B, T, vocab]
        next_token_logits = tuned_lens_logits[:, -1, :]     # [B, vocab]

        probs = next_token_logits.softmax(dim=-1)[0]
        next_token = torch.multinomial(probs, num_samples=1)

        scores.append(next_token_logits)
        prefix_tokens = torch.cat([prefix_tokens, next_token])

    layers.extend(reversed(removed_layers))

    return {
        "sequences": prefix_tokens[None],
        "scores": scores,
    }
