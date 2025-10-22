import numpy as np
from transformers import EvalPrediction
from dataclasses import dataclass
from typing import Literal, override

from model.reward import RewardModel
from .base import BaseRouter


class RewardRouter(BaseRouter):
    """Router that selects the LLM with the highest reward score."""

    def __init__(
        self,
        n_candidates: int,
        backbone_name_or_path: str,
        engine: Literal["huggingface", "vllm"] = "huggingface",
        **kwargs
    ):
        super().__init__(n_candidates)
        self.model = RewardModel(backbone_name_or_path, engine, **kwargs)  # type: ignore

    @override
    def route(
        self, prompts: str | list[str], responses: list[str] | list[list[str]] = [], **_
    ) -> EvalPrediction:
        assert len(prompts) == len(responses), "Number of prompts and responses must match."
        messages = [
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": r}]
            for prompt, response in zip(prompts, responses)
            for r in response
        ]
        scores = self.model.compute_reward(messages).view(-1, self.n_candidates)
        return EvalPrediction(predictions=scores.numpy(), label_ids=None)
