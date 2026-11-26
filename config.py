"""Configuration for the MMLU Ollama baseline."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPOSITORY_ROOT / "outputs"

DATASET_NAME = "cais/mmlu"
DATASET_CONFIG = "high_school_computer_science"
DATASET_SPLIT = "test"

MODELS = [
    "qwen2.5:1.5b",
    "gemma2:2b",
]

TEMPERATURE = 0.0
OLLAMA_API_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 120
