# LLM Representation Similarity

This repository is a shared starting point for studying whether language models
that make the same incorrect prediction have similar internal representations.
The current scope is a reproducible answer-collection baseline; hidden-state
extraction and CKA comparisons are future work.

## What the baseline does

The script loads the `test` split of the MMLU `high_school_computer_science`
configuration, sends every question to each configured local Ollama model using
the same prompt and temperature `0`, and saves per-model and combined CSV files.
Invalid answers and Ollama request failures are recorded as unanswered rather
than being counted as correct. Existing per-model CSVs in a chosen output
directory are reused, so completed models do not need to be rerun.

## Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running locally
- Enough memory to run the selected models

Install the Python dependencies:

```bash
git clone https://github.com/lucyrliu/NNDL.git
cd NNDL
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the default models through Ollama:

```bash
ollama pull qwen2.5:1.5b
ollama pull gemma2:2b
```

Make sure the Ollama service is running, then start the baseline:

```bash
python -m src.evaluate
```

The script downloads/caches MMLU through Hugging Face Datasets on its first run.
It prints progress for each model and question. Results go into a new timestamped
directory such as `outputs/baseline_YYYYMMDD_HHMMSS/`, so a new run does not
overwrite an earlier experiment.

To resume a run or use an explicit output directory, pass `--output-dir`. Any
existing, complete model CSVs in that directory will be reused:

```bash
python -m src.evaluate --output-dir outputs/my-baseline
```

If a model CSV already exists there, it is not overwritten; its question IDs,
model name, temperature, dataset, and prompt format are checked before reuse.

## Configuration

Edit [`config.py`](config.py) to change the model list, temperature, Ollama API
URL, request timeout, or dataset settings. Each model is queried with the same
prompt format, defined in `src/evaluate.py`. For a model not already available
locally, pull it first, for example:

```bash
ollama pull llama3.2:3b
```

Then add its exact Ollama model name to `MODELS` in `config.py`. Multiple models
can be added to the same experiment.

## Output files

Each run directory contains:

- `<model>_baseline.csv`: one row per question, including its ID, question,
  JSON-encoded choices, correct answer, predicted answer (A-D or blank), raw
  response, model name, correctness, request error if any, temperature, dataset
  configuration/split, and prompt format.
- `all_models_baseline_comparison.csv`: question ID, question, correct answer,
  each model's predicted answer, and a comparison category:
  `all_correct`, `same_incorrect`, `all_incorrect_different`, or
  `mixed_correctness`.

A blank predicted answer means the response did not contain a standalone A-D
answer or the request failed. It is always counted as incorrect. The comparison
category helps find questions where all models were correct, chose the same
wrong answer, were all wrong with different answers, or had mixed correctness.

## Tests

Run the focused tests without Ollama or a dataset download:

```bash
python -m unittest discover -s tests -v
```
