import json
import os

import torch


def _checkpoint_payload(
    step,
    candidate_layers,
    adapters,
    optimizer,
    model_name,
    seq_len,
    temperature,
    eval_size,
):
    return {
        "step": step,
        "candidate_layers": candidate_layers,
        "adapter_state_dict": adapters.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_name": model_name,
        "seq_len": seq_len,
        "temperature": temperature,
        "eval_size": eval_size,
    }


def save_adapter_checkpoint(
    out_dir,
    step,
    candidate_layers,
    adapters,
    optimizer,
    model_name,
    seq_len,
    temperature,
    eval_size,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"adapters_step_{step}.pt")

    torch.save(
        _checkpoint_payload(
            step=step,
            candidate_layers=candidate_layers,
            adapters=adapters,
            optimizer=optimizer,
            model_name=model_name,
            seq_len=seq_len,
            temperature=temperature,
            eval_size=eval_size,
        ),
        ckpt_path,
    )

    return ckpt_path


def save_final_adapters(
    out_dir,
    step,
    candidate_layers,
    adapters,
    optimizer,
    model_name,
    seq_len,
    temperature,
    eval_size,
):
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, "adapters_final.pt")

    torch.save(
        _checkpoint_payload(
            step=step,
            candidate_layers=candidate_layers,
            adapters=adapters,
            optimizer=optimizer,
            model_name=model_name,
            seq_len=seq_len,
            temperature=temperature,
            eval_size=eval_size,
        ),
        final_path,
    )

    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "candidate_layers": candidate_layers,
                "seq_len": seq_len,
                "temperature": temperature,
                "eval_size": eval_size,
                "final_step": step,
            },
            f,
            indent=2,
        )

    return final_path


def save_training_logs(logs_df, out_dir, filename="training_logs.jsonl"):
    os.makedirs(out_dir, exist_ok=True)
    logs_path = os.path.join(out_dir, filename)
    logs_df.to_json(logs_path, orient="records", lines=True)
    return logs_path
