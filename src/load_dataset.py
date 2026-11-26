"""Load the configured MMLU evaluation split."""


def load_questions(dataset_name, dataset_config, split):
    """Return dataset rows as dictionaries with stable sequential question IDs."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    return [
        {**dict(row), "question_id": index}
        for index, row in enumerate(dataset)
    ]
