import numpy as np
from typing import override
from transformers import EvalPrediction

from .base import BaseRouter


class FixedRouter(BaseRouter):
    """Router that always selects the same LLM for each query."""

    @override
    def __init__(self, n_candidates: int, target_index: int = 0, **_):
        """Initialize the FixedRouter.

        Args:
            n_candidates: Number of candidate LLMs.
            target_index: Index of the LLM to always select. Defaults to 0.
        """
        self.target_index = target_index
        super().__init__(n_candidates)

    @override
    def route(self, prompts: list[str], **_) -> EvalPrediction:
        predictions = np.zeros((len(prompts), self.n_candidates))
        predictions[:, self.target_index] = 1
        return EvalPrediction(predictions=predictions, label_ids=None)
