from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from tuned_lens.nn.lenses import TunedLens
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def resolve_hf_token(hf_token: Optional[str] = None) -> Optional[str]:
    """Resolve HF token without hardcoding secrets in source code."""
    return hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def load_model_tokenizer_and_lens(
    model_name: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase, TunedLens]:
    """Load a causal LM, tokenizer, and pretrained TunedLens."""
    token = resolve_hf_token(hf_token)
    device = torch.device(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=token,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tuned_lens = TunedLens.from_model_and_pretrained(model).to(device)
    tuned_lens.eval()

    return model, tokenizer, tuned_lens


def get_transformer_layers(model: torch.nn.Module):
    """Return the ModuleList of transformer blocks for supported model families."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    raise AttributeError(
        "Could not find transformer layers. Add this model family to "
        "early_exit.modeling.get_transformer_layers()."
    )
