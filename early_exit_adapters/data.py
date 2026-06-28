from datasets import load_dataset

from .preprocess import tokenize_example


def load_fineweb_stream(
    dataset_name="HuggingFaceFW/fineweb-edu",
    split="train",
    seed=42,
    buffer_size=10_000,
):
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=buffer_size,
    )

    return dataset


def build_fixed_eval_cache_from_stream(
    dataset,
    tokenizer,
    seq_len,
    eval_size,
):
    """
    Consume the first token-valid eval examples from the same streaming
    iterator that will later be used for training.
    """
    data_iter = iter(dataset)

    eval_batches = []
    raw_seen = 0

    while len(eval_batches) < eval_size:
        example = next(data_iter)
        raw_seen += 1

        batch = tokenize_example(
            example=example,
            tokenizer=tokenizer,
            seq_len=seq_len,
            device="cpu",
        )

        if batch is None:
            continue

        eval_batches.append(batch)

    print(
        f"Built fixed eval cache: {len(eval_batches)} examples "
        f"from {raw_seen} raw stream examples. "
        f"Training continues from the remaining stream."
    )

    return eval_batches, data_iter
