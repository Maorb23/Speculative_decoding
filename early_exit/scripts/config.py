from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EarlyExitConfig:
    model_name: str = "EleutherAI/pythia-1.4b-deduped"
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "CC-MAIN-2014-10"
    dataset_split: str = "train"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    gamma: int = 4
    seqlen: int = 10
    layer_index: int = 2
    num_prefix_tokens: int = 10
    num_examples: int = 100
    seed: int = 42
    hf_token: Optional[str] = None
    do_sample_vanilla: bool = True

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)
