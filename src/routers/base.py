from dataclasses import dataclass
from transformers import EvalPrediction


class BaseRouter:
    """Base class for all routers."""

    def __init__(self, n_candidates: int, **_):
        """Initialize the router.

        Args:
            n_candidates: Number of candidate LLMs.
        """
        self.n_candidates = n_candidates

    def route(self, prompts: list[str], **_) -> EvalPrediction:
        raise NotImplementedError()
