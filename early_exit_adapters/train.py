import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from .checkpointing import (
    save_adapter_checkpoint,
    save_final_adapters,
    save_training_logs,
)
from .data import build_fixed_eval_cache_from_stream
from .evaluate import evaluate_fixed_eval_set
from .logging_utils import (
    add_rows_to_wandb_log,
    build_eval_comparison_plots_for_wandb,
    eval_rows_from_metrics,
    metric_row,
)
from .losses import kl_distill_loss
from .metrics import compute_training_metrics
from .model import adapter_hidden_to_logits, build_adapters
from .preprocess import tokenize_example


def train_initial_adapters(
    model,
    tokenizer,
    dataset,
    candidate_layers,
    seq_len=128,
    max_steps=500,
    lr=1e-4,
    temperature=2.0,
    top_k=5,
    log_every=20,
    eval_step=100,
    eval_size=32,
    eval_max_batches=None,
    save_every=100,
    out_dir="data/early_exit_adapters",
    use_wandb=True,
    wandb_project="qwen35-early-exit-adapters",
    wandb_run_name=None,
    wandb_mode=None,
):
    os.makedirs(out_dir, exist_ok=True)

    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    model_name = getattr(model.config, "_name_or_path", None)

    candidate_layers = [layer for layer in candidate_layers if layer < num_layers]

    print("Training adapters for layers:", candidate_layers)

    adapters = build_adapters(
        model=model,
        candidate_layers=candidate_layers,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        adapters.parameters(),
        lr=lr,
        weight_decay=0.01,
    )
    wandb_run = None

    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            mode=wandb_mode,
            config={
                "seq_len": seq_len,
                "max_steps": max_steps,
                "lr": lr,
                "temperature": temperature,
                "top_k": top_k,
                "candidate_layers": candidate_layers,
                "log_every": log_every,
                "eval_step": eval_step,
                "eval_size": eval_size,
                "eval_max_batches": eval_max_batches,
                "save_every": save_every,
                "out_dir": out_dir,
                "model_name": model_name,
                "adapter_type": "ResidualLinearAdapter",
            },
        )

        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")

    eval_batches, train_iter = build_fixed_eval_cache_from_stream(
        dataset=dataset,
        tokenizer=tokenizer,
        seq_len=seq_len,
        eval_size=eval_size,
    )

    logs = []
    step = 0

    pbar = tqdm(total=max_steps, desc="Training adapters")

    for example in train_iter:
        if step >= max_steps:
            break

        batch = tokenize_example(
            example=example,
            tokenizer=tokenizer,
            seq_len=seq_len,
            device=device,
        )

        if batch is None:
            continue

        input_ids, attention_mask = batch

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            teacher_logits_full = outputs.logits.detach()
            hidden_states = outputs.hidden_states

        labels = input_ids[:, 1:]
        teacher_logits = teacher_logits_full[:, :-1, :]

        total_loss = 0.0
        layer_losses = {}

        for layer_index in candidate_layers:
            early_h = hidden_states[layer_index + 1].detach()
            early_h = early_h[:, :-1, :]

            student_logits = adapter_hidden_to_logits(
                model=model,
                adapter=adapters[str(layer_index)],
                hidden_state=early_h,
            )

            loss = kl_distill_loss(
                student_logits=student_logits.float(),
                teacher_logits=teacher_logits.float(),
                temperature=temperature,
            )

            total_loss = total_loss + loss
            layer_losses[layer_index] = loss

        total_loss = total_loss / len(candidate_layers)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=1.0)
        optimizer.step()

        do_train_log = step % log_every == 0
        do_eval = eval_step is not None and step % eval_step == 0

        if do_train_log or do_eval:
            log_record = {
                "step": step,
                "total_loss": float(total_loss.item()),
            }

            print("=" * 80)
            print(f"step {step} | total_loss={total_loss.item():.4f}")

            wandb_log = (
                {
                    "step": step,
                    "train/total_loss": float(total_loss.item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                }
                if use_wandb
                else None
            )

            if do_train_log:
                train_rows = []

                for layer in candidate_layers:
                    with torch.no_grad():
                        early_h = hidden_states[layer + 1].detach()[:, :-1, :]

                        student_logits = adapter_hidden_to_logits(
                            model=model,
                            adapter=adapters[str(layer)],
                            hidden_state=early_h,
                        )

                        metrics = compute_training_metrics(
                            student_logits=student_logits.float(),
                            teacher_logits=teacher_logits.float(),
                            labels=labels,
                            top_k=top_k,
                        )

                    layer_loss = float(layer_losses[layer].item())
                    row = metric_row(layer, metrics, top_k, kl_loss=layer_loss)
                    train_rows.append(row)

                    print(
                        f"train layer {layer:02d} | "
                        f"kl_loss={row['kl_loss']:.4f} | "
                        f"metric_kl={row['metric_kl']:.4f} | "
                        f"ce={row['ce']:.4f} | "
                        f"top1={row['top1']:.3f} | "
                        f"accept_exact={row['accept_exact']:.3f}"
                    )

                    for key, value in metrics.items():
                        log_record[f"layer_{layer}/train/{key}"] = value
                    log_record[f"layer_{layer}/train/kl_loss"] = layer_loss

                if use_wandb:
                    add_rows_to_wandb_log(wandb_log, step, "train", train_rows)

            if do_eval:
                print("-" * 80)
                print("running fixed held-out eval...")

                eval_metrics = evaluate_fixed_eval_set(
                    model=model,
                    adapters=adapters,
                    eval_batches=eval_batches,
                    candidate_layers=candidate_layers,
                    top_k=top_k,
                    max_eval_batches=eval_max_batches,
                )

                log_record.update(eval_metrics)

                baseline_rows, adapted_rows, _ = eval_rows_from_metrics(
                    eval_metrics=eval_metrics,
                    candidate_layers=candidate_layers,
                    top_k=top_k,
                )

                for baseline_row, adapted_row in zip(baseline_rows, adapted_rows):
                    layer = baseline_row["layer"]

                    print(
                        f"eval layer {layer:02d} | "
                        f"KL {baseline_row['metric_kl']:.4f} -> "
                        f"{adapted_row['metric_kl']:.4f} | "
                        f"accept {baseline_row['accept_exact']:.3f} -> "
                        f"{adapted_row['accept_exact']:.3f} | "
                        f"top1 {baseline_row['top1']:.3f} -> "
                        f"{adapted_row['top1']:.3f}"
                    )

                if use_wandb:
                    wandb_log.update(
                        build_eval_comparison_plots_for_wandb(
                            step=step,
                            baseline_rows=baseline_rows,
                            adapted_rows=adapted_rows,
                        )
                    )

            if use_wandb:
                import wandb

                wandb.log(wandb_log, step=step)

            logs.append(log_record)

        if step > 0 and step % save_every == 0:
            ckpt_path = save_adapter_checkpoint(
                out_dir=out_dir,
                step=step,
                candidate_layers=candidate_layers,
                adapters=adapters,
                optimizer=optimizer,
                model_name=model_name,
                seq_len=seq_len,
                temperature=temperature,
                eval_size=eval_size,
            )

            print("saved:", ckpt_path)

        step += 1
        pbar.update(1)

    pbar.close()

    final_path = save_final_adapters(
        out_dir=out_dir,
        step=step,
        candidate_layers=candidate_layers,
        adapters=adapters,
        optimizer=optimizer,
        model_name=model_name,
        seq_len=seq_len,
        temperature=temperature,
        eval_size=eval_size,
    )

    logs_df = pd.DataFrame(logs)
    logs_path = save_training_logs(logs_df, out_dir=out_dir)

    print("saved final adapters:", final_path)
    print("saved logs:", logs_path)

    if wandb_run is not None:
        wandb_run.finish()

    return adapters, logs_df
