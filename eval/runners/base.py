"""Abstract Runner base class — every provider runner subclasses this.

A ``Runner`` knows how to take an :class:`~eval.schemas.MCQItem` and produce
an :class:`~eval.schemas.EvalResponse`. Concrete runners live in
``eval.runners.<provider>_runner`` (see ``anthropic_runner`` for the
reference implementation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from eval.schemas import EvalResponse, MCQItem


class Runner(ABC):
    """Abstract provider runner.

    Subclasses set ``name`` (a short provider tag like ``"anthropic"``) at the
    class level and implement :meth:`run`. ``model`` is the exact model id
    string passed to the SDK; ``temperature`` and ``top_p`` are sampling
    parameters propagated to every API call.
    """

    name: str = "base"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        """Initialize the runner with sampling parameters.

        Args:
            model: Exact model id passed to the provider's SDK.
            temperature: Sampling temperature (default 0.0 for determinism).
            top_p: Nucleus-sampling cutoff (default 1.0).
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def run(self, item: MCQItem) -> EvalResponse:
        """Run a single item through the model.

        Implementations must:

        * Format ``item`` via ``eval.prompts.mcq.format_mcq``.
        * Send the prompt to the provider with the runner's sampling params.
        * Retry transient failures with exponential backoff.
        * On final failure, return an :class:`EvalResponse` whose ``error``
          field is set; never raise.

        Args:
            item: The MCQ to evaluate.

        Returns:
            A populated :class:`EvalResponse`. ``parsed_answer`` may be
            ``None`` if the model's text response could not be parsed
            into the expected structured format.
        """
