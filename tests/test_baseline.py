import csv
import tempfile
import unittest
from pathlib import Path
from requests.exceptions import HTTPError
from unittest.mock import Mock, patch

from src import compare_answers
from src.evaluate import (
    answer_letter,
    build_prompt,
    evaluate_model,
    extract_answer,
    load_existing_results,
    run_evaluation,
    write_model_results,
)
from src.ollama_client import OllamaError, generate


QUESTION = {
    "question_id": 0,
    "question": "Which option is correct?",
    "choices": ["first", "second", "third", "fourth"],
    "answer": 2,
}


class AnswerParsingTests(unittest.TestCase):
    def test_extracts_standalone_answer_letter(self):
        self.assertEqual(extract_answer("The answer is C."), "C")
        self.assertEqual(extract_answer("I cannot decide."), "")

    def test_converts_mmlu_answer_to_letter(self):
        self.assertEqual(answer_letter(0), "A")
        self.assertEqual(answer_letter(3), "D")
        self.assertEqual(answer_letter("b"), "B")
        with self.assertRaises(ValueError):
            answer_letter(4)

    def test_prompt_contains_all_choices(self):
        prompt = build_prompt(QUESTION)
        self.assertIn("A. first", prompt)
        self.assertIn("D. fourth", prompt)
        self.assertIn("Answer with only the option letter", prompt)


class OllamaClientTests(unittest.TestCase):
    @patch("src.ollama_client.requests.post")
    def test_sends_non_streaming_request_with_temperature(self, post):
        response = Mock()
        response.json.return_value = {"response": " B "}
        post.return_value = response

        answer = generate(
            "example:1b", "prompt", temperature=0, api_url="http://local/api",
            timeout=10,
        )

        self.assertEqual(answer, "B")
        post.assert_called_once_with(
            "http://local/api",
            json={
                "model": "example:1b",
                "prompt": "prompt",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=10,
        )

    @patch("src.ollama_client.requests.post")
    def test_raises_clear_error_for_http_failure(self, post):
        response = Mock()
        response.raise_for_status.side_effect = HTTPError("HTTP failure")
        post.return_value = response
        with self.assertRaises(OllamaError):
            generate("example:1b", "prompt")


class EvaluationTests(unittest.TestCase):
    @patch("src.evaluate.generate", return_value="not an option")
    def test_invalid_response_is_not_correct(self, _generate):
        rows = evaluate_model(
            "example:1b", [QUESTION], 0, "http://local/api", 1, progress=lambda _: None
        )
        self.assertEqual(rows[0]["model_answer"], "")
        self.assertFalse(rows[0]["is_correct"])

    @patch("src.evaluate.generate", side_effect=OllamaError("request timed out"))
    def test_request_failure_is_recorded_as_incorrect(self, _generate):
        rows = evaluate_model(
            "example:1b", [QUESTION], 0, "http://local/api", 1, progress=lambda _: None
        )
        self.assertEqual(rows[0]["model_answer"], "")
        self.assertIn("timed out", rows[0]["error"])
        self.assertFalse(rows[0]["is_correct"])

    def test_existing_results_are_validated_and_loaded(self):
        rows = evaluate_model(
            "example:1b",
            [QUESTION],
            0,
            "http://local/api",
            1,
            progress=lambda _: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.csv"
            write_model_results(path, rows)
            loaded = load_existing_results(path, "example:1b", [QUESTION], 0)
            self.assertEqual(loaded[0]["question_id"], "0")
            self.assertEqual(loaded[0]["is_correct"], False)
            with self.assertRaises(FileExistsError):
                write_model_results(path, rows)

    def test_comparison_categories(self):
        self.assertEqual(
            compare_answers.comparison_category("A", ["A", "A"]), "all_correct"
        )
        self.assertEqual(
            compare_answers.comparison_category("A", ["B", "B"]), "same_incorrect"
        )
        self.assertEqual(
            compare_answers.comparison_category("A", ["B", "C"]),
            "all_incorrect_different",
        )
        self.assertEqual(
            compare_answers.comparison_category("A", ["A", "B"]),
            "mixed_correctness",
        )
        self.assertEqual(
            compare_answers.comparison_category("A", ["", ""]),
            "all_incorrect_different",
        )

    def test_existing_comparison_must_match_expected_results(self):
        rows = [{
            "question_id": 0,
            "question": "Question",
            "correct_answer": "A",
            "model": "A",
            "comparison": "all_correct",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            compare_answers.write_comparison_csv(path, rows, ["model"])
            self.assertTrue(
                compare_answers.existing_comparison_matches(path, rows, ["model"])
            )
            self.assertFalse(
                compare_answers.existing_comparison_matches(
                    path, [{**rows[0], "model": "B"}], ["model"]
                )
            )

    @patch("src.evaluate.load_questions", return_value=[{
        "question_id": 99,
        "question": QUESTION["question"],
        "choices": QUESTION["choices"],
        "answer": QUESTION["answer"],
    }])
    @patch("src.evaluate.generate", return_value="C")
    def test_run_writes_model_and_combined_csv(self, _generate, _load_questions):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_evaluation(["example:1b"], directory)
            model_path = run_dir / "example_1b_baseline.csv"
            comparison_path = run_dir / "all_models_baseline_comparison.csv"
            with model_path.open(newline="", encoding="utf-8") as file:
                model_row = next(csv.DictReader(file))
            with comparison_path.open(newline="", encoding="utf-8") as file:
                comparison_row = next(csv.DictReader(file))
            rerun_dir = run_evaluation(["example:1b"], directory)

        self.assertEqual(model_row["question_id"], "0")
        self.assertEqual(model_row["model_answer"], "C")
        self.assertEqual(model_row["is_correct"], "True")
        self.assertEqual(comparison_row["example:1b"], "C")
        self.assertEqual(comparison_row["comparison"], "all_correct")
        self.assertEqual(rerun_dir, run_dir)
        _generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
