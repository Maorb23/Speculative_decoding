"""Early-exit speculative decoding utilities."""

from .config import EarlyExitConfig
from .modeling import load_model_tokenizer_and_lens
from .draft import generate_with_tuned_lens
from .spec_decode import speculative_decode_once, speculative_generate
from .benchmark import run_benchmark

__all__ = [
    "EarlyExitConfig",
    "load_model_tokenizer_and_lens",
    "generate_with_tuned_lens",
    "speculative_decode_once",
    "speculative_generate",
    "run_benchmark",
]
