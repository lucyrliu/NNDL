"""Run the MMLU multiple-choice baseline with local Ollama models."""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import config
from src import compare_answers
from src.load_dataset import load_questions
from src.ollama_client import OllamaError, generate

PROMPT_FORMAT = "mmlu_multiple_choice_v1"
ANSWER_LETTERS = ("A", "B", "C", "D")
MODEL_FILE_SUFFIX = "_baseline.csv"


def answer_letter(answer):
    """Convert an MMLU answer index or letter into A-D."""
    if isinstance(answer, bool):
        raise ValueError(f"Invalid MMLU answer: {answer!r}")
    if isinstance(answer, int) and 0 <= answer < len(ANSWER_LETTERS):
        return ANSWER_LETTERS[answer]
    if isinstance(answer, str):
        normalized = answer.strip().upper()
        if normalized in ANSWER_LETTERS:
            return normalized
        if normalized in ("0", "1", "2", "3"):
            return ANSWER_LETTERS[int(normalized)]
    raise ValueError(f"Invalid MMLU answer: {answer!r}")


def extract_answer(response):
    """Extract the first standalone A-D option letter, or return an empty string."""
    match = re.search(r"\b([A-D])\b", response.upper())
    return match.group(1) if match else ""


def build_prompt(question):
    """Format one MMLU row using the shared baseline prompt."""
    choices = question["choices"]
    if len(choices) != 4:
        raise ValueError(
            f"Question {question['question_id']} must have exactly four choices."
        )
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(ANSWER_LETTERS, choices)
    )
    return (
        "Answer this multiple-choice question.\n\n"
        f"Question: {question['question']}\n"
        f"Options:\n{options}\n\n"
        "Answer with only the option letter (A, B, C, or D)."
    )


def safe_model_name(model):
    """Create a filesystem-safe name from an Ollama model identifier."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._")
    if not name:
        raise ValueError(f"Model name {model!r} cannot be used as a filename.")
    return name


def model_result_path(output_dir, model):
    return output_dir / f"{safe_model_name(model)}{MODEL_FILE_SUFFIX}"


def evaluate_model(model, questions, temperature, api_url, timeout, progress=print):
    """Query one model for each question and return its result rows."""
    rows = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        progress(f"  Question {index}/{total}")
        raw_response = ""
        error = ""
        try:
            raw_response = generate(
                model,
                build_prompt(question),
                temperature=temperature,
                api_url=api_url,
                timeout=timeout,
            )
        except OllamaError as exc:
            error = str(exc)
            progress(f"    Request error: {error}")

        correct = answer_letter(question["answer"])
        prediction = extract_answer(raw_response)
        rows.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "choices": json.dumps(question["choices"], ensure_ascii=False),
                "correct_answer": correct,
                "model_answer": prediction,
                "raw_response": raw_response,
                "model": model,
                "is_correct": prediction == correct,
                "error": error,
                "temperature": temperature,
                "dataset": config.DATASET_NAME,
                "dataset_config": config.DATASET_CONFIG,
                "dataset_split": config.DATASET_SPLIT,
                "prompt_format": PROMPT_FORMAT,
            }
        )
    return rows


def load_existing_results(path, model, questions, temperature):
    """Read an existing complete model CSV without rerunning that model."""
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    expected_ids = [str(question["question_id"]) for question in questions]
    actual_ids = [row.get("question_id", "") for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Existing results at {path} do not match the current question IDs."
        )
    if any(
        row.get("model") != model
        or row.get("temperature") != str(temperature)
        or row.get("dataset") != config.DATASET_NAME
        or row.get("dataset_config") != config.DATASET_CONFIG
        or row.get("dataset_split") != config.DATASET_SPLIT
        or row.get("prompt_format") != PROMPT_FORMAT
        for row in rows
    ):
        raise ValueError(
            f"Existing results at {path} use different experiment settings."
        )
    for row in rows:
        row["is_correct"] = row["is_correct"].lower() == "true"
    return rows


def write_model_results(path, rows):
    """Write per-question model results."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {path}")
    fieldnames = [
        "question_id",
        "question",
        "choices",
        "correct_answer",
        "model_answer",
        "raw_response",
        "model",
        "is_correct",
        "error",
        "temperature",
        "dataset",
        "dataset_config",
        "dataset_split",
        "prompt_format",
    ]
    with path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_output_dir(output_dir=None):
    """Create a unique timestamped run directory, or use an explicit directory."""
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.OUTPUTS_DIR / f"baseline_{timestamp}"
    suffix = 1
    while path.exists():
        path = config.OUTPUTS_DIR / f"baseline_{timestamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def run_evaluation(models, output_dir=None):
    """Run or reuse results for all configured models and create comparison CSV."""
    if not models:
        raise ValueError("Configure at least one Ollama model in config.py.")
    run_dir = create_output_dir(output_dir)
    questions = load_questions(
        config.DATASET_NAME, config.DATASET_CONFIG, config.DATASET_SPLIT
    )
    normalized_questions = [
        {
            "question_id": index,
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "correct_answer": answer_letter(row["answer"]),
        }
        for index, row in enumerate(questions)
    ]

    results_by_model = {}
    for model in models:
        path = model_result_path(run_dir, model)
        print(f"\nRunning {model} at temperature {config.TEMPERATURE}...")
        if path.exists():
            print(f"  Reusing existing results: {path}")
            rows = load_existing_results(
                path, model, normalized_questions, config.TEMPERATURE
            )
        else:
            rows = evaluate_model(
                model,
                normalized_questions,
                config.TEMPERATURE,
                config.OLLAMA_API_URL,
                config.REQUEST_TIMEOUT_SECONDS,
            )
            write_model_results(path, rows)
            correct_count = sum(row["is_correct"] for row in rows)
            accuracy = correct_count / len(rows) if rows else 0
            print(f"  Saved {path} ({accuracy:.1%} accuracy)")
        results_by_model[model] = rows

    comparison_path = run_dir / "all_models_baseline_comparison.csv"
    comparison_rows = compare_answers.build_comparison_rows(
        normalized_questions, results_by_model
    )
    if comparison_path.exists():
        if not compare_answers.existing_comparison_matches(
            comparison_path, comparison_rows, list(results_by_model)
        ):
            raise ValueError(
                f"Existing comparison at {comparison_path} does not match these "
                "results. Choose a new --output-dir to preserve both experiments."
            )
        print(f"  Preserving existing comparison: {comparison_path}")
    else:
        compare_answers.write_comparison_csv(
            comparison_path, comparison_rows, list(results_by_model)
        )
        print(f"  Saved comparison: {comparison_path}")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="Use this directory to resume/reuse results; otherwise create a new run directory.",
    )
    args = parser.parse_args(argv)
    try:
        output_dir = run_evaluation(config.MODELS, args.output_dir)
    except (OllamaError, ValueError, OSError, RuntimeError) as exc:
        print(f"Baseline evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nBaseline evaluation complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
