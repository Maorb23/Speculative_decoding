import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .metrics import compute_training_metrics
from .model import adapter_hidden_to_logits
from .preprocess import tokenize_example


@torch.no_grad()
def evaluate_fixed_eval_set(
    model,
    adapters,
    eval_batches,
    candidate_layers,
    top_k=5,
    max_eval_batches=None,
):
    """Evaluate baseline and adapted logits on a fixed cached eval set."""
    device = next(model.parameters()).device

    was_training = adapters.training
    adapters.eval()

    if max_eval_batches is not None:
        eval_batches_to_use = eval_batches[:max_eval_batches]
    else:
        eval_batches_to_use = eval_batches

    sums = {}
    counts = {}

    for input_ids_cpu, attention_mask_cpu in eval_batches_to_use:
        input_ids = input_ids_cpu.to(device)
        attention_mask = attention_mask_cpu.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        teacher_logits = outputs.logits[:, :-1, :].detach()
        labels = input_ids[:, 1:]
        hidden_states = outputs.hidden_states

        for layer_index in candidate_layers:
            early_h = hidden_states[layer_index + 1].detach()
            early_h = early_h[:, :-1, :]

            baseline_h = model.model.norm(early_h)
            baseline_logits = model.lm_head(baseline_h)

            baseline_metrics = compute_training_metrics(
                student_logits=baseline_logits.float(),
                teacher_logits=teacher_logits.float(),
                labels=labels,
                top_k=top_k,
            )

            adapted_logits = adapter_hidden_to_logits(
                model=model,
                adapter=adapters[str(layer_index)],
                hidden_state=early_h,
            )

            adapted_metrics = compute_training_metrics(
                student_logits=adapted_logits.float(),
                teacher_logits=teacher_logits.float(),
                labels=labels,
                top_k=top_k,
            )

            for metric_name, value in baseline_metrics.items():
                key = f"layer_{layer_index}/eval_baseline/{metric_name}"
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

            for metric_name, value in adapted_metrics.items():
                key = f"layer_{layer_index}/eval_adapted/{metric_name}"
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

    eval_results = {key: sums[key] / counts[key] for key in sums}

    if was_training:
        adapters.train()

    return eval_results


@torch.no_grad()
def layers_kl_over_data_rows(
    model,
    tokenizer,
    dataset,
    candidate_layers,
    adapters=None,
    run_name="baseline",
    seq_len=128,
    num_eval_rows=100,
    temperature=2.0,
    top_k=5,
    out_dir="data/layer_kl_eval",
    save_path=None,
):
    """
    Compare early-layer logits to final model logits.

    Baseline: early_h -> final_norm -> lm_head.
    Adapted:  early_h -> adapter -> final_norm -> lm_head.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    if adapters is not None:
        adapters.eval()

    num_layers = len(model.model.layers)
    candidate_layers = [layer for layer in candidate_layers if layer < num_layers]

    if save_path is None:
        save_path = f"{run_name}_layer_kl.jsonl"

    print(f"{run_name} KL eval")
    print("num_layers:", num_layers)
    print("candidate_layers:", candidate_layers)

    records = []
    valid_rows = 0
    seen_rows = 0

    pbar = tqdm(total=num_eval_rows, desc=f"{run_name} layer KL")

    for example in dataset:
        if valid_rows >= num_eval_rows:
            break

        seen_rows += 1

        batch = tokenize_example(
            example=example,
            tokenizer=tokenizer,
            seq_len=seq_len,
            device=device,
        )

        if batch is None:
            continue

        input_ids, attention_mask = batch

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        teacher_logits = outputs.logits[:, :-1, :].detach().float()
        hidden_states = outputs.hidden_states
        labels = input_ids[:, 1:]

        for layer_index in candidate_layers:
            early_h = hidden_states[layer_index + 1].detach()
            early_h = early_h[:, :-1, :]

            if adapters is None:
                student_h = early_h
            else:
                adapter = adapters[str(layer_index)]
                adapter_dtype = next(adapter.parameters()).dtype
                student_h = adapter(early_h.to(dtype=adapter_dtype)).to(
                    dtype=model_dtype
                )

            student_logits = model.lm_head(model.model.norm(student_h)).float()

            metrics = compute_training_metrics(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                top_k=top_k,
            )
            metrics["kl_to_teacher"] = float(
                (
                    F.kl_div(
                        F.log_softmax(student_logits / temperature, dim=-1),
                        F.softmax(teacher_logits / temperature, dim=-1),
                        reduction="batchmean",
                    )
                    * (temperature * temperature)
                ).item()
            )

            record = {
                "run_name": run_name,
                "valid_row_index": valid_rows,
                "seen_row_index": seen_rows,
                "layer_index": layer_index,
                "seq_len": seq_len,
                "n_tokens": int(labels.numel()),
                "temperature": temperature,
            }
            record.update(metrics)
            records.append(record)

        valid_rows += 1
        pbar.update(1)

    pbar.close()

    records_df = pd.DataFrame(records)

    output_path = os.path.join(out_dir, save_path)
    records_df.to_json(output_path, orient="records", lines=True)

    summary = (
        records_df.groupby(["run_name", "layer_index"])
        .agg(
            rows=("valid_row_index", "count"),
            n_tokens=("n_tokens", "sum"),
            kl_to_teacher=("kl_to_teacher", "mean"),
            ce=("ce", "mean"),
            perplexity=("perplexity", "mean"),
            mean_gt_prob=("mean_gt_prob", "mean"),
            top1_teacher_agreement=("top1_teacher_agreement", "mean"),
            topk_overlap=(f"top{top_k}_overlap", "mean"),
            accept_proxy_exact=("accept_proxy_exact", "mean"),
            accept_proxy_sampled=("accept_proxy_sampled", "mean"),
        )
        .reset_index()
    )

    summary_path = os.path.join(out_dir, f"{run_name}_layer_kl_summary.json")
    summary.to_json(summary_path, orient="records", indent=2)

    print("saved records:", output_path)
    print("saved summary:", summary_path)

    return records_df, summary
