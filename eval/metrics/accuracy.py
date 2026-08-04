"""Top-1 accuracy on MCQ items.

For non-adversarial items (``correct`` is set), the response is correct iff
``parsed_answer == correct`` (case-insensitive). For adversarial items
(``correct`` is ``None``), the response is correct iff the model abstained
(``parsed_answer is None``); picking any letter on an adversarial item
counts as wrong.

Formula:

    accuracy = n_correct / n

where ``n`` is the number of responses that match a ground-truth item by
``item_id``. Responses without a matching ground-truth item are skipped
and not counted toward ``n``.
"""

from __future__ import annotations

from collections.abc import Iterable

from eval.schemas import EvalResponse, MCQItem


def top1_accuracy(
    responses: Iterable[EvalResponse],
    ground_truth: Iterable[MCQItem],
) -> dict[str, float]:
    """Compute top-1 accuracy on a parallel iterable of responses + items.

    Args:
        responses: Iterable of model responses; matched to ground truth
            by ``item_id``.
        ground_truth: Iterable of MCQ items with the answer key.

    Returns:
        ``{"n": int, "n_correct": int, "accuracy": float}``. ``accuracy``
        is 0.0 when there are no matched responses.
    """
    by_item: dict[str, MCQItem] = {item.id: item for item in ground_truth}
    n = 0
    n_correct = 0
    for resp in responses:
        truth = by_item.get(resp.item_id)
        if truth is None:
            continue
        n += 1
        if _is_correct(resp, truth):
            n_correct += 1
    accuracy = n_correct / n if n > 0 else 0.0
    return {"n": float(n), "n_correct": float(n_correct), "accuracy": accuracy}


def _is_correct(resp: EvalResponse, truth: MCQItem) -> bool:
    """Decide whether a single response matches the truth.

    Adversarial items (``truth.correct is None``) require the model to
    abstain (``resp.parsed_answer is None``). Otherwise the parsed letter
    must match the ground-truth letter case-insensitively.
    """
    if truth.correct is None:
        return resp.parsed_answer is None
    if resp.parsed_answer is None:
        return False
    return resp.parsed_answer.upper() == truth.correct.upper()
