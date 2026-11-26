"""Small HTTP client for Ollama's generate endpoint."""

import requests


class OllamaError(RuntimeError):
    """An Ollama request failed or returned an invalid response."""


def generate(
    model,
    prompt,
    temperature=0.0,
    api_url="http://localhost:11434/api/generate",
    timeout=120,
):
    """Return Ollama's response text, raising OllamaError on request failures."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        response_data = response.json()
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:
        raise OllamaError(f"Ollama returned invalid JSON: {exc}") from exc

    answer = response_data.get("response") if isinstance(response_data, dict) else None
    if not isinstance(answer, str):
        raise OllamaError("Ollama response did not contain a text 'response' field.")
    return answer.strip()
