import numpy as np
from typing import override
from transformers import EvalPrediction

from .base import BaseRouter


class RandomRouter(BaseRouter):
    """Router that randomly selects target LLMs for each query."""

    @override
    def route(self, prompts: list[str], **_) -> EvalPrediction:
        return EvalPrediction(predictions=np.random.rand(len(prompts), self.n_candidates), label_ids=None)
