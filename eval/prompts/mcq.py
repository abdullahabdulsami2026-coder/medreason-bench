"""Multiple-choice prompt template.

Renders an :class:`~eval.schemas.MCQItem` into the prompt string a runner
sends to the model, with the confidence-elicitation suffix appended.
"""

from __future__ import annotations

import hashlib

from eval.prompts.confidence import CONFIDENCE_SUFFIX
from eval.schemas import MCQItem

MCQ_TEMPLATE: str = (
    "You will answer a clinical multiple-choice question.\n\n"
    "Stem:\n{stem}\n\n"
    "Question: {question}\n\n"
    "Options:\n{options_block}\n\n"
    "{confidence_suffix}"
)


def format_mcq(item: MCQItem) -> str:
    """Render an :class:`~eval.schemas.MCQItem` into the runner's prompt string.

    Args:
        item: The MCQ item to format.

    Returns:
        A fully-formed user-message string with stem, question, options,
        and the confidence-elicitation suffix.
    """
    options_block = "\n".join(f"{letter}. {text}" for letter, text in sorted(item.options.items()))
    return MCQ_TEMPLATE.format(
        stem=item.stem.strip(),
        question=item.question.strip(),
        options_block=options_block,
        confidence_suffix=CONFIDENCE_SUFFIX,
    )


def prompt_hash(item: MCQItem) -> str:
    """Stable SHA-256 of the rendered prompt for the run manifest.

    Used to detect prompt drift across runs: if the rendered prompt
    text changes for the same item id, the hash changes.
    """
    return hashlib.sha256(format_mcq(item).encode("utf-8")).hexdigest()
