"""Utilities for training early-exit residual adapters."""

__all__ = [
    "ResidualLinearAdapter",
    "adapter_hidden_to_logits",
    "build_adapters",
    "load_lm_model_and_tokenizer",
]


def __getattr__(name):
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
