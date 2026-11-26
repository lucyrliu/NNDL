"""Build a question-aligned comparison across model result sets."""

import csv


def comparison_category(correct_answer, predictions):
    """Describe the correctness agreement for one question."""
    correctness = [
        prediction == correct_answer
        for prediction in predictions
    ]
    if all(correctness):
        return "all_correct"
    if not any(correctness):
        if predictions and predictions[0] and len(set(predictions)) == 1:
            return "same_incorrect"
        return "all_incorrect_different"
    return "mixed_correctness"


def build_comparison_rows(questions, results_by_model):
    """Join model predictions by question ID and derive a comparison category."""
    models = list(results_by_model)
    results_by_id = {}
    for model, rows in results_by_model.items():
        indexed_rows = {}
        for row in rows:
            question_id = int(row["question_id"])
            if question_id in indexed_rows:
                raise ValueError(f"Duplicate question ID {question_id} for {model}.")
            indexed_rows[question_id] = row
        results_by_id[model] = indexed_rows

    comparison_rows = []
    for question in questions:
        question_id = int(question["question_id"])
        correct_answer = question["correct_answer"]
        row = {
            "question_id": question_id,
            "question": question["question"],
            "correct_answer": correct_answer,
        }
        predictions = []
        for model in models:
            model_row = results_by_id[model].get(question_id)
            if model_row is None:
                raise ValueError(
                    f"Missing question ID {question_id} in results for {model}."
                )
            if model_row["correct_answer"] != correct_answer:
                raise ValueError(
                    f"Correct answer mismatch for question {question_id} in {model}."
                )
            prediction = model_row["model_answer"] or ""
            row[model] = prediction
            predictions.append(prediction)
        row["comparison"] = comparison_category(correct_answer, predictions)
        comparison_rows.append(row)
    return comparison_rows


def write_comparison_csv(path, rows, models):
    """Write the combined comparison CSV."""
    fieldnames = [
        "question_id",
        "question",
        "correct_answer",
        *models,
        "comparison",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_comparison_matches(path, rows, models):
    """Check that an existing comparison is identical to the expected output."""
    fieldnames = [
        "question_id",
        "question",
        "correct_answer",
        *models,
        "comparison",
    ]
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != fieldnames:
            return False
        existing_rows = list(reader)
    expected_rows = [
        {field: str(row[field]) for field in fieldnames}
        for row in rows
    ]
    return existing_rows == expected_rows
