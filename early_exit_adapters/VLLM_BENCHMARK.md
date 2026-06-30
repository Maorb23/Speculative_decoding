# vLLM Speculative Decoding Benchmark

This is the next speed path after the HF/PyTorch diagnostic experiments.
Start with vLLM's built-in small draft model support, then use those results
as the baseline for custom truncated/adapted draft work.

## Install

Install vLLM in the GPU environment used for benchmarking. vLLM is usually run
on Linux with CUDA, so this may belong in Kaggle, a cloud GPU VM, or WSL rather
than the local Windows workspace.

```bash
pip install vllm
```

Set a Hugging Face token if either model requires one:

```bash
export HF_TOKEN=...
```

The script also accepts `HUGGINGFACE_TOKEN` and mirrors it to `HF_TOKEN`.

## Built-In Small-Draft Benchmark

Run target-only and speculative decoding back to back:

```bash
python -m early_exit_adapters.vllm_speculative_benchmark \
  --target-model Qwen/Qwen3.5-2B \
  --draft-model Qwen/Qwen3.5-0.8B \
  --mode both \
  --num-speculative-tokens 4 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 \
  --max-tokens 128 \
  --repeats 3 \
  --out results/vllm_qwen35_2b_08b.json
```

The `max_model_len` cap is important for Qwen3.5: vLLM can otherwise resolve
the model's context length to a very large value, which is not appropriate for a
small T4 benchmark.

For lower peak memory or cleaner allocator behavior, run the two modes in
separate processes:

```bash
python -m early_exit_adapters.vllm_speculative_benchmark \
  --mode target \
  --max-model-len 4096 \
  --out results/vllm_target_only.json

python -m early_exit_adapters.vllm_speculative_benchmark \
  --mode speculative \
  --max-model-len 4096 \
  --out results/vllm_small_draft.json
```

## Outputs

The JSON report includes:

- `tokens_per_sec`
- `mean_latency_sec`
- per-run latency and output token counts
- CUDA peak allocated/reserved memory when `torch.cuda` can report it
- vLLM speculative counters when exposed:
  - `num_drafts`
  - `num_draft_tokens`
  - `num_accepted_tokens`
  - `acceptance_rate`
  - `mean_acceptance_length`
  - `acceptance_by_position`

## Custom Draft Next Steps

The current script intentionally uses vLLM's normal `draft_model` path first.
For the early-exit model, the desired draft computation is:

```text
truncated Qwen layers 0..k -> optional adapter[k] -> target norm/head -> logits
```

vLLM does not accept arbitrary PyTorch hooks as the draft engine in this script.
The next integration options are:

1. Package the truncated Qwen draft as a standalone HF-compatible causal LM.
2. Add the adapter into that exported draft model's forward path.
3. If export is not feasible, implement a custom vLLM model/plugin and compare
   it against the small-model draft benchmark above.
