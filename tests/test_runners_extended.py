"""Tests for the Phase-4 runners (Google, Groq, OpenAI, Ollama).

Each runner is exercised via an injected mock ``client=``; the SDKs
themselves are never called over the network. Tests cover happy path,
text extraction, parsing edges, full retry exhaustion, transient
recovery, and sampling-parameter propagation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from eval.runners.google_runner import GoogleRunner
from eval.runners.groq_runner import GroqRunner
from eval.runners.ollama_runner import OllamaRunner
from eval.runners.openai_runner import OpenAIRunner
from eval.schemas import MCQItem

SAMPLE_ITEM = MCQItem(
    id="t1",
    source="test",
    stem="A 65-year-old presents with chest pain.",
    question="What is the most appropriate first step?",
    options={"A": "ECG", "B": "CT angiogram", "C": "Echo", "D": "Stress test"},
    correct="A",
)

CLEAN_TEXT = (
    "ANSWER: A\nCONFIDENCE: 90\nRATIONALE: ECG is the appropriate first-line investigation."
)


# ── helpers — mock-client builders for each provider's response shape ──


def _gemini_client(text: str) -> Any:
    """Mock ``google.genai.Client`` with ``client.models.generate_content``."""
    client = MagicMock()
    msg = MagicMock()
    msg.text = text
    client.models.generate_content.return_value = msg
    return client


def _openai_compat_client(text: str) -> Any:
    """Mock OpenAI / Groq client with ``chat.completions.create``."""
    client = MagicMock()
    msg = MagicMock()
    msg.choices = [MagicMock(message=MagicMock(content=text))]
    client.chat.completions.create.return_value = msg
    return client


def _ollama_client_pydantic(text: str) -> Any:
    """Mock Ollama client returning a Pydantic-like ``ChatResponse``."""
    client = MagicMock()
    msg = MagicMock()
    msg.message = MagicMock(content=text)
    client.chat.return_value = msg
    return client


def _ollama_client_dict(text: str) -> Any:
    """Mock Ollama client returning a legacy dict response."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": text}}
    return client


# ── GoogleRunner ────────────────────────────────────────────────


def test_google_runner_parses_clean_response() -> None:
    runner = GoogleRunner(model="test-model", client=_gemini_client(CLEAN_TEXT), max_retries=1)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.confidence == 90
    assert resp.rationale is not None and "ECG" in resp.rationale
    assert resp.error is None
    assert resp.raw_response == CLEAN_TEXT


def test_google_runner_returns_error_after_retries() -> None:
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("API down")
    runner = GoogleRunner(model="test-model", client=client, max_retries=3, initial_backoff_s=0.0)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer is None
    assert resp.error is not None and "API down" in resp.error
    assert client.models.generate_content.call_count == 3


def test_google_runner_recovers_after_transient_failure() -> None:
    msg = MagicMock(text=CLEAN_TEXT)
    client = MagicMock()
    client.models.generate_content.side_effect = [RuntimeError("transient"), msg]
    runner = GoogleRunner(model="test-model", client=client, max_retries=3, initial_backoff_s=0.0)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.error is None
    assert client.models.generate_content.call_count == 2


def test_google_runner_passes_sampling_params_to_sdk() -> None:
    client = _gemini_client(CLEAN_TEXT)
    runner = GoogleRunner(
        model="m",
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
        client=client,
        max_retries=1,
    )
    runner.run(SAMPLE_ITEM)
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "m"
    assert "config" in kwargs


def test_google_runner_falls_back_to_candidates_when_text_missing() -> None:
    """When ``msg.text`` is empty, the helper walks candidates → content → parts."""
    client = MagicMock()
    msg = MagicMock()
    msg.text = ""
    msg.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(text=CLEAN_TEXT)])),
    ]
    client.models.generate_content.return_value = msg
    runner = GoogleRunner(model="m", client=client, max_retries=1)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"


# ── GroqRunner ──────────────────────────────────────────────────


def test_groq_runner_parses_clean_response() -> None:
    runner = GroqRunner(model="test-model", client=_openai_compat_client(CLEAN_TEXT), max_retries=1)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.confidence == 90
    assert resp.error is None


def test_groq_runner_returns_error_after_retries() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API down")
    runner = GroqRunner(model="test-model", client=client, max_retries=3, initial_backoff_s=0.0)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer is None
    assert resp.error is not None and "API down" in resp.error
    assert client.chat.completions.create.call_count == 3


def test_groq_runner_recovers_after_transient_failure() -> None:
    msg = MagicMock()
    msg.choices = [MagicMock(message=MagicMock(content=CLEAN_TEXT))]
    client = MagicMock()
    client.chat.completions.create.side_effect = [RuntimeError("transient"), msg]
    runner = GroqRunner(model="test-model", client=client, max_retries=3, initial_backoff_s=0.0)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.error is None


def test_groq_runner_passes_sampling_params_to_sdk() -> None:
    client = _openai_compat_client(CLEAN_TEXT)
    runner = GroqRunner(
        model="m",
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
        client=client,
        max_retries=1,
    )
    runner.run(SAMPLE_ITEM)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_tokens"] == 512
    assert kwargs["messages"][0]["role"] == "user"


# ── OpenAIRunner ────────────────────────────────────────────────


def test_openai_runner_parses_clean_response() -> None:
    runner = OpenAIRunner(
        model="test-model", client=_openai_compat_client(CLEAN_TEXT), max_retries=1
    )
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.confidence == 90
    assert resp.error is None


def test_openai_runner_returns_error_after_retries() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("rate limit")
    runner = OpenAIRunner(model="test-model", client=client, max_retries=3, initial_backoff_s=0.0)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer is None
    assert resp.error is not None and "rate limit" in resp.error
    assert client.chat.completions.create.call_count == 3


def test_openai_runner_passes_sampling_params_to_sdk() -> None:
    client = _openai_compat_client(CLEAN_TEXT)
    runner = OpenAIRunner(
        model="gpt-4o",
        temperature=0.5,
        top_p=0.95,
        max_tokens=2048,
        client=client,
        max_retries=1,
    )
    runner.run(SAMPLE_ITEM)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 2048


# ── OllamaRunner ────────────────────────────────────────────────


def test_ollama_runner_parses_pydantic_response() -> None:
    runner = OllamaRunner(
        model="test-model", client=_ollama_client_pydantic(CLEAN_TEXT), max_retries=1
    )
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.confidence == 90
    assert resp.error is None


def test_ollama_runner_parses_dict_response() -> None:
    """Backward compatibility with older ollama SDK versions."""
    runner = OllamaRunner(model="test-model", client=_ollama_client_dict(CLEAN_TEXT), max_retries=1)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.error is None


def test_ollama_runner_returns_error_after_retries() -> None:
    client = MagicMock()
    client.chat.side_effect = ConnectionError("daemon offline")
    runner = OllamaRunner(model="test-model", client=client, max_retries=3, initial_backoff_s=0.0)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer is None
    assert resp.error is not None and "daemon offline" in resp.error


def test_ollama_runner_passes_sampling_params_to_sdk() -> None:
    client = _ollama_client_pydantic(CLEAN_TEXT)
    runner = OllamaRunner(
        model="llama3.1:8b",
        temperature=0.2,
        top_p=0.85,
        max_tokens=256,
        client=client,
        max_retries=1,
    )
    runner.run(SAMPLE_ITEM)
    kwargs = client.chat.call_args.kwargs
    assert kwargs["model"] == "llama3.1:8b"
    assert kwargs["options"]["temperature"] == 0.2
    assert kwargs["options"]["top_p"] == 0.85
    assert kwargs["options"]["num_predict"] == 256


# ── Registry registration ───────────────────────────────────────


def test_runner_registry_contains_all_phase4_providers() -> None:
    """All Phase-4A runners must be registered in the pipeline registry."""
    from eval.pipeline import RUNNER_REGISTRY

    assert "anthropic" in RUNNER_REGISTRY
    assert "google" in RUNNER_REGISTRY
    assert "groq" in RUNNER_REGISTRY
    assert "openai" in RUNNER_REGISTRY
    assert "ollama" in RUNNER_REGISTRY
    assert RUNNER_REGISTRY["google"] is GoogleRunner
    assert RUNNER_REGISTRY["groq"] is GroqRunner
    assert RUNNER_REGISTRY["openai"] is OpenAIRunner
    assert RUNNER_REGISTRY["ollama"] is OllamaRunner
