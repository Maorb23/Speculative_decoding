# Early-Exit TunedLens Speculative Decoding

Modular version of `shot_16_speculative_decoding_made_easy_with_only_one_model.ipynb`.

## Install

```bash
pip install torch transformers datasets tuned-lens numpy
```

Set your Hugging Face token as an environment variable instead of hardcoding it:

```bash
export HF_TOKEN="hf_..."
```

## Run

From the parent directory of `early_exit/`:

```bash
python -m early_exit.run_benchmark \
  --model-name EleutherAI/pythia-1.4b-deduped \
  --device cuda \
  --gamma 4 \
  --layer-index 2 \
  --seqlen 10 \
  --num-prefix-tokens 10 \
  --num-examples 100 \
  --out results/early_exit_benchmark.json
```

## Files

- `config.py` - benchmark configuration dataclass.
- `modeling.py` - model/tokenizer/TunedLens loading and layer access helpers.
- `data.py` - streaming dataset loader.
- `draft.py` - early-layer TunedLens draft generation.
- `spec_decode.py` - speculative acceptance/rejection logic.
- `benchmark.py` - benchmark loop.
- `run_benchmark.py` - CLI entry point.

## Important notes

This keeps the notebook's original mechanism: it temporarily removes later transformer layers, drafts tokens with the TunedLens at an early layer, then restores the full model for verification.

This is useful pedagogically, but for serious speed measurements you should avoid popping/restoring layers each step and instead build a separate shallow draft model once.
