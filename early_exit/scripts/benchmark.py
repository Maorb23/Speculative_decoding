from __future__ import annotations

import time
from typing import Dict, Iterable, List

import numpy as np
import torch
import transformers

from .config import EarlyExitConfig
from .data import load_streaming_text_dataset
from .modeling import load_model_tokenizer_and_lens
from .spec_decode import speculative_generate


@torch.no_grad()
def benchmark_one_example(
    model: torch.nn.Module,
    tokenizer,
    tuned_lens: torch.nn.Module,
    text: str,
    config: EarlyExitConfig,
) -> Dict[str, float]:
    device = config.torch_device

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)[0][: config.num_prefix_tokens]
    if input_ids.numel() == 0:
        return {"skip": 1.0}

    start = time.time()
    spec_out = speculative_generate(
        model=model,
        tuned_lens=tuned_lens,
        input_ids=input_ids,
        seqlen=config.seqlen,
        layer_index=config.layer_index,
        gamma=config.gamma,
    )
    speculative_time = time.time() - start

    num_new_tokens = spec_out["sequences"].shape[1] - input_ids.shape[0]

    attention_mask = torch.ones_like(input_ids[None], device=device)
    start = time.time()
    model.generate(
        input_ids[None],
        attention_mask=attention_mask,
        max_new_tokens=num_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=config.do_sample_vanilla,
    )
    vanilla_time = time.time() - start

    stats = spec_out["stats"]
    return {
        "skip": 0.0,
        "accepted": float(stats.accepted),
        "generated": float(stats.generated),
        "alpha": stats.alpha,
        "speculative_time": speculative_time,
        "vanilla_time": vanilla_time,
        "improvement": 1.0 - speculative_time / vanilla_time if vanilla_time > 0 else float("nan"),
    }


def run_benchmark(config: EarlyExitConfig) -> List[Dict[str, float]]:
    transformers.set_seed(config.seed)

    model, tokenizer, tuned_lens = load_model_tokenizer_and_lens(
        model_name=config.model_name,
        device=config.torch_device,
        dtype=config.dtype,
        hf_token=config.hf_token,
    )

    ds = load_streaming_text_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
    )

    results: List[Dict[str, float]] = []
    total_accepted = 0.0
    total_generated = 0.0
    spec_times: List[float] = []
    vanilla_times: List[float] = []

    with torch.no_grad():
        for idx, example in enumerate(ds):
            if idx >= config.num_examples:
                break

            result = benchmark_one_example(
                model=model,
                tokenizer=tokenizer,
                tuned_lens=tuned_lens,
                text=example["text"],
                config=config,
            )
            if result.get("skip", 0.0):
                continue

            results.append(result)
            total_accepted += result["accepted"]
            total_generated += result["generated"]
            spec_times.append(result["speculative_time"])
            vanilla_times.append(result["vanilla_time"])

            running_alpha = total_accepted / total_generated if total_generated else 0.0
            running_improvement = 1.0 - np.mean(spec_times) / np.mean(vanilla_times)
            print(
                f"example={len(results)} "
                f"alpha={running_alpha:.2%} "
                f"improvement={running_improvement:.4f}"
            )

    return results
