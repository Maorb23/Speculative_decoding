import argparse
import json
from pathlib import Path

import torch

from early_exit.config import EarlyExitConfig
from early_exit.benchmark import run_benchmark


def parse_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    parser = argparse.ArgumentParser(description="Early-exit TunedLens speculative decoding benchmark")
    parser.add_argument("--model-name", default="EleutherAI/pythia-1.4b-deduped")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=10)
    parser.add_argument("--layer-index", type=int, default=2)
    parser.add_argument("--num-prefix-tokens", type=int, default=10)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=None, help="Prefer setting HF_TOKEN env var instead.")
    parser.add_argument("--out", default=None, help="Optional JSON path for benchmark results.")
    args = parser.parse_args()

    config = EarlyExitConfig(
        model_name=args.model_name,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        gamma=args.gamma,
        seqlen=args.seqlen,
        layer_index=args.layer_index,
        num_prefix_tokens=args.num_prefix_tokens,
        num_examples=args.num_examples,
        seed=args.seed,
        hf_token=args.hf_token,
    )

    results = run_benchmark(config)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
