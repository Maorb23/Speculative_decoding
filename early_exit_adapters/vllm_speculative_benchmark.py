"""Benchmark vLLM target-only and draft-model speculative decoding.

This is the first production-oriented speed path after the PyTorch/HF
diagnostic experiments. It intentionally starts with vLLM's built-in
``draft_model`` speculative decoding before attempting custom truncated or
adapter-backed draft models.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "Explain speculative decoding in three concise paragraphs.",
    "Write a short Python function that computes top-k token overlap.",
    "Summarize why KV caching matters for autoregressive decoding.",
    "Give practical advice for benchmarking draft and target language models.",
]


@dataclass
class BenchmarkConfig:
    target_model: str
    draft_model: str
    mode: str
    num_speculative_tokens: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int | None
    max_num_seqs: int | None
    enforce_eager: bool
    enable_chunked_prefill: bool
    trust_remote_code: bool
    dtype: str
    repeats: int
    warmup_runs: int


def _read_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS

    prompt_path = Path(path)
    suffix = prompt_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                raise ValueError("JSON prompt files must contain a list of strings.")
            return data
        if isinstance(data, dict) and isinstance(data.get("prompts"), list):
            prompts = data["prompts"]
            if not all(isinstance(item, str) for item in prompts):
                raise ValueError("'prompts' must be a list of strings.")
            return prompts
        raise ValueError("JSON prompt files must be a list or an object with 'prompts'.")

    return [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cuda_memory_snapshot() -> dict[str, float | None]:
    try:
        import torch
    except ImportError:
        return {
            "cuda_peak_allocated_gb": None,
            "cuda_peak_reserved_gb": None,
            "cuda_free_gb": None,
            "cuda_total_gb": None,
        }

    if not torch.cuda.is_available():
        return {
            "cuda_peak_allocated_gb": None,
            "cuda_peak_reserved_gb": None,
            "cuda_free_gb": None,
            "cuda_total_gb": None,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gb = 1024**3
    return {
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / gb,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / gb,
        "cuda_free_gb": free_bytes / gb,
        "cuda_total_gb": total_bytes / gb,
    }


def _reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cleanup_engine(llm: Any) -> None:
    del llm
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _metric_value(metric: Any, default: float = 0.0) -> float:
    value = getattr(metric, "value", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_values(metric: Any) -> list[float]:
    values = getattr(metric, "values", None)
    if values is None:
        return []
    return [float(value) for value in values]


def _collect_speculative_metrics(llm: Any, num_speculative_tokens: int) -> dict[str, Any]:
    metrics = llm.get_metrics()
    num_drafts = 0.0
    num_draft_tokens = 0.0
    num_accepted_tokens = 0.0
    accepted_tokens_per_pos = [0.0] * num_speculative_tokens

    for metric in metrics:
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_drafts":
            num_drafts += _metric_value(metric)
        elif name == "vllm:spec_decode_num_draft_tokens":
            num_draft_tokens += _metric_value(metric)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted_tokens += _metric_value(metric)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for pos, value in enumerate(_metric_values(metric)):
                if pos < len(accepted_tokens_per_pos):
                    accepted_tokens_per_pos[pos] += value

    acceptance_by_position = [
        value / num_drafts if num_drafts else None
        for value in accepted_tokens_per_pos
    ]

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_rate": (
            num_accepted_tokens / num_draft_tokens if num_draft_tokens else None
        ),
        "mean_acceptance_length": (
            1.0 + num_accepted_tokens / num_drafts if num_drafts else None
        ),
        "acceptance_by_position": acceptance_by_position,
    }


def _build_llm(config: BenchmarkConfig, mode: str) -> Any:
    from vllm import LLM

    speculative_config = None
    if mode == "speculative":
        speculative_config = {
            "method": "draft_model",
            "model": config.draft_model,
            "num_speculative_tokens": config.num_speculative_tokens,
            "enforce_eager": config.enforce_eager,
        }
        if config.max_model_len is not None:
            speculative_config["max_model_len"] = config.max_model_len

    kwargs: dict[str, Any] = {
        "model": config.target_model,
        "trust_remote_code": config.trust_remote_code,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enforce_eager": config.enforce_eager,
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "speculative_config": speculative_config,
        "disable_log_stats": False,
    }
    if config.dtype != "auto":
        kwargs["dtype"] = config.dtype
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    if config.max_num_seqs is not None:
        kwargs["max_num_seqs"] = config.max_num_seqs

    return LLM(**kwargs)


def run_vllm_benchmark(
    config: BenchmarkConfig,
    prompts: list[str],
    mode: str,
) -> dict[str, Any]:
    from vllm import SamplingParams

    llm = _build_llm(config, mode)
    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )

    for _ in range(config.warmup_runs):
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    _reset_cuda_peak_memory()
    run_records = []
    last_outputs = None
    for run_idx in range(config.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        elapsed_sec = time.perf_counter() - start
        output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        run_records.append(
            {
                "run_idx": run_idx,
                "latency_sec": elapsed_sec,
                "output_tokens": output_tokens,
                "tokens_per_sec": output_tokens / elapsed_sec if elapsed_sec else None,
            }
        )
        last_outputs = outputs

    total_tokens = sum(record["output_tokens"] for record in run_records)
    total_latency = sum(record["latency_sec"] for record in run_records)
    result = {
        "mode": mode,
        "target_model": config.target_model,
        "draft_model": config.draft_model if mode == "speculative" else None,
        "num_prompts": len(prompts),
        "num_speculative_tokens": (
            config.num_speculative_tokens if mode == "speculative" else None
        ),
        "total_latency_sec": total_latency,
        "total_output_tokens": total_tokens,
        "tokens_per_sec": total_tokens / total_latency if total_latency else None,
        "mean_latency_sec": total_latency / len(run_records) if run_records else None,
        "runs": run_records,
        "memory": _cuda_memory_snapshot(),
    }

    if mode == "speculative":
        result["speculative_metrics"] = _collect_speculative_metrics(
            llm,
            config.num_speculative_tokens,
        )

    if last_outputs:
        result["sample_outputs"] = [
            {
                "prompt": prompts[idx],
                "text": output.outputs[0].text,
                "output_tokens": len(output.outputs[0].token_ids),
            }
            for idx, output in enumerate(last_outputs[: min(2, len(last_outputs))])
        ]

    _cleanup_engine(llm)
    return result


def compare_results(results: dict[str, Any]) -> dict[str, Any] | None:
    baseline = results.get("target")
    speculative = results.get("speculative")
    if not baseline or not speculative:
        return None

    baseline_tps = baseline.get("tokens_per_sec")
    speculative_tps = speculative.get("tokens_per_sec")
    baseline_latency = baseline.get("mean_latency_sec")
    speculative_latency = speculative.get("mean_latency_sec")

    return {
        "tokens_per_sec_speedup": (
            speculative_tps / baseline_tps
            if baseline_tps and speculative_tps
            else None
        ),
        "latency_reduction": (
            1.0 - speculative_latency / baseline_latency
            if baseline_latency and speculative_latency
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM target-only and draft-model speculative decoding."
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--mode",
        choices=["target", "speculative", "both"],
        default="both",
        help="Run target-only, speculative decoding, or both sequentially.",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _read_prompts(args.prompts_file)
    if not prompts:
        raise ValueError("At least one prompt is required.")

    config = BenchmarkConfig(
        target_model=args.target_model,
        draft_model=args.draft_model,
        mode=args.mode,
        num_speculative_tokens=args.num_speculative_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        trust_remote_code=not args.no_trust_remote_code,
        dtype=args.dtype,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
    )

    if os.environ.get("HF_TOKEN") is None and os.environ.get("HUGGINGFACE_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    modes = ["target", "speculative"] if args.mode == "both" else [args.mode]
    results: dict[str, Any] = {
        "config": asdict(config),
        "prompt_count": len(prompts),
        "target": None,
        "speculative": None,
        "comparison": None,
    }

    for mode in modes:
        result = run_vllm_benchmark(config, prompts, mode)
        results[mode] = result
        print(json.dumps(result, indent=2))

    results["comparison"] = compare_results(results)
    if results["comparison"] is not None:
        print(json.dumps({"comparison": results["comparison"]}, indent=2))

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote benchmark results to {out_path}")


if __name__ == "__main__":
    main()
