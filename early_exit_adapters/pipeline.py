import argparse

from .data import load_fineweb_stream
from .hf_utils import save_hf_metadata, upload_adapter_folder_to_hf
from .model import load_lm_model_and_tokenizer
from .train import train_initial_adapters


def main(
    model_name="Qwen/Qwen3.5-2B",
    dataset_name="HuggingFaceFW/fineweb-edu",
    candidate_layers=None,
    seq_len=128,
    max_steps=500,
    lr=1e-4,
    temperature=2.0,
    top_k=5,
    log_every=20,
    eval_step=100,
    eval_size=32,
    save_every=100,
    out_dir="data/early_exit_adapters",
    use_wandb=True,
    wandb_project="qwen35-early-exit-adapters",
    wandb_run_name="qwen35_2b_residual_adapters",
    wandb_mode=None,
    upload_to_hf=False,
    hf_repo_id=None,
):
    if candidate_layers is None:
        candidate_layers = [4, 8, 12, 16, 18, 20]

    model, tokenizer, _ = load_lm_model_and_tokenizer(model_name=model_name)

    dataset = load_fineweb_stream(
        dataset_name=dataset_name,
        split="train",
        seed=42,
        buffer_size=10_000,
    )

    adapters, logs_df = train_initial_adapters(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        candidate_layers=candidate_layers,
        seq_len=seq_len,
        max_steps=max_steps,
        lr=lr,
        temperature=temperature,
        top_k=top_k,
        log_every=log_every,
        eval_step=eval_step,
        eval_size=eval_size,
        save_every=save_every,
        out_dir=out_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
    )

    save_hf_metadata(
        folder_path=out_dir,
        model_name=model_name,
        candidate_layers=candidate_layers,
        seq_len=seq_len,
        temperature=temperature,
        extra_metadata={"eval_size": eval_size},
    )

    if upload_to_hf:
        if not hf_repo_id:
            raise ValueError("hf_repo_id is required when upload_to_hf=True")
        upload_adapter_folder_to_hf(
            repo_id=hf_repo_id,
            folder_path=out_dir,
        )

    return adapters, logs_df


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--candidate-layers", nargs="+", type=int, default=[4, 8, 12, 16, 18, 20])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-step", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--out-dir", default="data/early_exit_adapters")
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="qwen35-early-exit-adapters")
    parser.add_argument("--wandb-run-name", default="qwen35_2b_residual_adapters")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--upload-to-hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf-repo-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        candidate_layers=args.candidate_layers,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        lr=args.lr,
        temperature=args.temperature,
        top_k=args.top_k,
        log_every=args.log_every,
        eval_step=args.eval_step,
        eval_size=args.eval_size,
        save_every=args.save_every,
        out_dir=args.out_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        upload_to_hf=args.upload_to_hf,
        hf_repo_id=args.hf_repo_id,
    )
