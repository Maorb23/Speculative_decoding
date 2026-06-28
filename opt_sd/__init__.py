"""OPT-Tree speculative decoding utilities.

This package currently contains the tree-based two-model speculative decoder
under ``opt_sd.scripts.sd_tree`` plus a report visualizer. Older early-exit
imports used to live here, but those modules are not present in this package.
"""

from .scripts.sd_tree import (
    construct_opt_tree,
    forward_target_model_on_tree,
    loading_models,
    opt_tree_speculative_decoding_step,
    speculative_decode_tree,
    verify_tree_and_sample_output,
)
from .scripts.tree_visualizer import visualize_opt_tree_run, visualize_tree_run

__all__ = [
    "construct_opt_tree",
    "forward_target_model_on_tree",
    "loading_models",
    "opt_tree_speculative_decoding_step",
    "speculative_decode_tree",
    "verify_tree_and_sample_output",
    "visualize_opt_tree_run",
    "visualize_tree_run",
]
