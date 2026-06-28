"""Utilities for training early-exit residual adapters."""

from .model import (
    ResidualLinearAdapter,
    adapter_hidden_to_logits,
    build_adapters,
    load_lm_model_and_tokenizer,
)

__all__ = [
    "ResidualLinearAdapter",
    "adapter_hidden_to_logits",
    "build_adapters",
    "load_lm_model_and_tokenizer",
]
