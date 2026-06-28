import torch


def tokenize_example(example, tokenizer, seq_len, device):
    text = example.get("text", "")

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
    )

    input_ids = encoded["input_ids"]

    if input_ids.shape[1] < seq_len:
        return None

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    return input_ids, attention_mask
