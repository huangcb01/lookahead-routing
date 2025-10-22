import numpy as np
from typing import override
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from transformers import EvalPrediction

from args import DataArguments
from data.dataset import load_train_datasets
from utils.math import mean
from .base import BaseRouter


class KNNRouter(BaseRouter):
    @override
    def __init__(
        self,
        n_candidates: int,
        k: int,
        data_dir: str,
        train_datasets: list[str],
        candidate_llms: list[str],
        embedder: str = "all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 128,
        **_,
    ):
        super().__init__(n_candidates)
        self.embedder = SentenceTransformer(embedder, device=device)
        self.train_data = load_train_datasets(
            DataArguments(
                data_dir=data_dir,
                train_datasets=train_datasets,
                candidate_llms=candidate_llms,
                score_normalization="min-max",
            )
        )[0]
        self.train_data_embeddings = self.embedder.encode(
            [item["prompt"] for item in self.train_data],
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        self.k = k
        self.batch_size = batch_size

    @override
    def route(self, prompts: list[str], **_) -> EvalPrediction:
        prompt_embeddings = self.embedder.encode(
            prompts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=True
        )
        hits = semantic_search(
            prompt_embeddings,
            self.train_data_embeddings,
            top_k=self.k,
            query_chunk_size=self.batch_size,
            score_function=dot_score,
        )
        predictions = [
            [
                mean([self.train_data[h["corpus_id"]]["scores"][i] for h in hit])
                            for i in range(self.n_candidates)
            ]
            for hit in hits
        ]
        return EvalPrediction(predictions=np.array(predictions), label_ids=None)
