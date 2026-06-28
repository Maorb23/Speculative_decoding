import json
import os

import torch

from .model import build_adapters


def load_env(env_path=".env"):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    return load_dotenv(env_path)


def _hf_token(token=None):
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def create_or_get_hf_repo(repo_id, token=None, private=False):
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token(token))
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    return repo_id


def upload_adapter_folder_to_hf(repo_id, folder_path, token=None, private=False):
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token(token))
    create_or_get_hf_repo(repo_id=repo_id, token=token, private=private)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder_path,
    )


def download_adapter_checkpoint_from_hf(repo_id, filename, token=None, cache_dir=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=_hf_token(token),
        cache_dir=cache_dir,
    )


def load_adapters_from_hf(
    model,
    repo_id,
    filename="adapters_final.pt",
    device=None,
    token=None,
    cache_dir=None,
):
    if device is None:
        device = next(model.parameters()).device

    ckpt_path = download_adapter_checkpoint_from_hf(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=cache_dir,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    candidate_layers = ckpt["candidate_layers"]

    adapters = build_adapters(
        model=model,
        candidate_layers=candidate_layers,
        device=device,
    )
    adapters.load_state_dict(ckpt["adapter_state_dict"])
    adapters.eval()

    return adapters


def save_hf_metadata(
    folder_path,
    model_name,
    candidate_layers,
    seq_len,
    temperature,
    extra_metadata=None,
):
    os.makedirs(folder_path, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "candidate_layers": candidate_layers,
        "seq_len": seq_len,
        "temperature": temperature,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = os.path.join(folder_path, "hf_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme_path = os.path.join(folder_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "# Early Exit Adapters\n\n"
            "Adapters trained to map intermediate hidden states to "
            "final-model-like next-token distributions.\n\n"
            f"Base model: {model_name}\n\n"
            f"Layers: {candidate_layers}\n\n"
            f"Sequence length: {seq_len}\n\n"
            f"Temperature: {temperature}\n"
        )

    return metadata_path
