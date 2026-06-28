import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResidualLinearAdapter(nn.Module):
    """Residual adapter: h -> h + Linear(LayerNorm(h))."""

    def __init__(self, hidden_size):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, h):
        return h + self.proj(self.ln(h))


def build_adapters(model, candidate_layers, device):
    hidden_size = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype

    adapters = nn.ModuleDict(
        {
            str(layer): ResidualLinearAdapter(hidden_size)
            for layer in candidate_layers
        }
    )

    adapters.to(device=device, dtype=model_dtype)
    return adapters


def load_lm_model_and_tokenizer(model_name, device=None, dtype=None):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=hf_token,
        trust_remote_code=True,
    ).to(device)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer, device


def adapter_hidden_to_logits(model, adapter, hidden_state):
    """early hidden -> adapter -> final norm -> lm_head."""
    adapted_h = adapter(hidden_state)
    adapted_h = model.model.norm(adapted_h)
    logits = model.lm_head(adapted_h)
    return logits
