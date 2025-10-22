import torch
import numpy as np
from typing import override
from sentence_transformers import SentenceTransformer
from transformers import EvalPrediction
from sklearn.cluster import KMeans

from args import DataArguments
from data.dataset import load_train_datasets
from .base import BaseRouter


class HardClusterRouter(BaseRouter):
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
        self.k = k
        self.batch_size = batch_size

        # Embed all prompts in the training data
        self.train_data_embeddings = self.embedder.encode(
            [item["prompt"] for item in self.train_data],
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        # Cluster
        self.clusterer = KMeans(n_clusters=self.k, random_state=0).fit(self.train_data_embeddings.cpu().numpy())
        self.cluster_centers = torch.tensor(self.clusterer.cluster_centers_, device=self.train_data_embeddings.device)
        labels = self.clusterer.labels_
        self.cluster_scores = [
            np.array([self.train_data[j]["scores"] for j in np.where(labels == i)[0].tolist()]).mean(axis=0)
            for i in range(self.k)
        ]

    @override
    def route(self, prompts: list[str], **_) -> EvalPrediction:
        prompt_embeddings = self.embedder.encode(
            prompts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=True
        )
        cluster = self.clusterer.predict(prompt_embeddings.cpu().numpy())
        predictions = [self.cluster_scores[i] for i in cluster]
        return EvalPrediction(predictions=predictions, label_ids=None)
