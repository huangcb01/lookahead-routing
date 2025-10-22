import numpy as np
from typing import override
from transformers import EvalPrediction

from utils import argmax
from .base import BaseRouter


class OracleRouter(BaseRouter):
    """Router that selects the LLM with the highest score."""

    @override
    def route(
        self, prompts: list[str], scores: list[list[float]] = [], **_
    ) -> EvalPrediction:
        assert len(prompts) == len(scores), "Number of prompts and scores must be equal."
        return EvalPrediction(predictions=np.array(scores), label_ids=None)
