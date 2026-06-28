from datasets import load_dataset


def load_streaming_text_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str = "CC-MAIN-2014-10",
    split: str = "train",
):
    """Load the streaming text dataset used by the original notebook."""
    return load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
    )
