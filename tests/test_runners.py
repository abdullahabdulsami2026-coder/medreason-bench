"""Tests for ``eval.runners.anthropic_runner.AnthropicRunner``.

The Anthropic SDK is never called over the network from these tests — every
test injects a mock client through the runner's ``client=`` constructor arg.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from eval.prompts.confidence import parse_confidence_response
from eval.runners.anthropic_runner import AnthropicRunner
from eval.schemas import MCQItem

SAMPLE_ITEM = MCQItem(
    id="t1",
    source="test",
    stem="A 65-year-old presents with chest pain.",
    question="What is the most appropriate first step?",
    options={"A": "ECG", "B": "CT angiogram", "C": "Echo", "D": "Stress test"},
    correct="A",
)


def make_mock_client(text_response: str) -> Any:
    """Build a MagicMock that mimics ``anthropic.Anthropic`` for happy paths."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=text_response)]
    client.messages.create.return_value = msg
    return client


# ── Happy path ────────────────────────────────────────────────────


def test_runner_parses_clean_response() -> None:
    text = "ANSWER: A\nCONFIDENCE: 90\nRATIONALE: ECG is the appropriate first-line investigation."
    runner = AnthropicRunner(
        model="test-model",
        client=make_mock_client(text),
        max_retries=1,
    )
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "A"
    assert resp.confidence == 90
    assert resp.rationale is not None
    assert "ECG" in resp.rationale
    assert resp.error is None
    assert resp.model == "test-model"
    assert resp.item_id == SAMPLE_ITEM.id
    assert resp.raw_response == text


def test_runner_extracts_text_from_multi_block_response() -> None:
    """msg.content is a list of typed content blocks; runner concatenates text."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = [
        MagicMock(text="ANSWER: B\n"),
        MagicMock(text="CONFIDENCE: 60\n"),
        MagicMock(text="RATIONALE: combined."),
    ]
    client.messages.create.return_value = msg
    runner = AnthropicRunner(model="m", client=client, max_retries=1)
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "B"
    assert resp.confidence == 60


# ── Parsing edge cases (confidence parser) ────────────────────────


def test_confidence_parser_clamps_to_0_100() -> None:
    a, c, r = parse_confidence_response("ANSWER: A\nCONFIDENCE: 150\nRATIONALE: x")
    assert a == "A"
    assert c == 100


def test_confidence_parser_returns_none_when_format_violated() -> None:
    a, c, r = parse_confidence_response("I think the answer is A.")
    assert a is None
    assert c is None
    assert r is None


def test_confidence_parser_handles_lowercase_letter() -> None:
    a, _, _ = parse_confidence_response("ANSWER: a\nCONFIDENCE: 50\nRATIONALE: x")
    assert a == "A"


# ── Failure / retry behaviour ─────────────────────────────────────


def test_runner_returns_error_response_after_retries() -> None:
    """All attempts fail → no exception; an EvalResponse with error= is returned."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("API down")
    runner = AnthropicRunner(
        model="test-model",
        client=client,
        max_retries=3,
        initial_backoff_s=0.0,  # no real sleeps in tests
    )
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer is None
    assert resp.confidence is None
    assert resp.error is not None
    assert "API down" in resp.error
    assert client.messages.create.call_count == 3


def test_runner_recovers_after_transient_failure() -> None:
    """First call fails, second succeeds → response is parsed normally."""
    text = "ANSWER: D\nCONFIDENCE: 80\nRATIONALE: stress is contraindicated acutely."
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    client = MagicMock()
    client.messages.create.side_effect = [RuntimeError("transient"), msg]
    runner = AnthropicRunner(
        model="test-model",
        client=client,
        max_retries=3,
        initial_backoff_s=0.0,
    )
    resp = runner.run(SAMPLE_ITEM)
    assert resp.parsed_answer == "D"
    assert resp.confidence == 80
    assert resp.error is None
    assert client.messages.create.call_count == 2


# ── Sampling-parameter propagation ────────────────────────────────


def test_runner_passes_sampling_params_to_sdk() -> None:
    client = make_mock_client("ANSWER: A\nCONFIDENCE: 50\nRATIONALE: x")
    runner = AnthropicRunner(
        model="m",
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
        client=client,
        max_retries=1,
    )
    runner.run(SAMPLE_ITEM)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_tokens"] == 512
    assert kwargs["messages"][0]["role"] == "user"
