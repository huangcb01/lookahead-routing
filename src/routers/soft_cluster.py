import torch
import numpy as np
from typing import override
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from transformers import EvalPrediction
from sklearn.cluster import KMeans

from args import DataArguments
from data.dataset import load_train_datasets
from utils.math import mean
from .base import BaseRouter


class SoftClusterRouter(BaseRouter):
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
        clusterer = KMeans(n_clusters=self.k, random_state=0).fit(self.train_data_embeddings.cpu().numpy())
        self.cluster_centers = torch.tensor(clusterer.cluster_centers_, device=self.train_data_embeddings.device)
        labels = clusterer.labels_
        self.cluster_scores = []
        for i in range(self.k):
            cluster_indices = np.where(labels == i)[0]
            embeddings = self.train_data_embeddings[cluster_indices]
            scores = torch.tensor(
                [self.train_data[j]["scores"] for j in cluster_indices.tolist()],
                device=self.train_data_embeddings.device,
            )
            distances = torch.cdist(embeddings, self.cluster_centers[i].unsqueeze(0))
            weights = torch.softmax(-distances, dim=0)
            self.cluster_scores.append((weights * scores).sum(dim=0))
        self.cluster_scores = torch.stack(self.cluster_scores)

    @override
    def route(self, prompts: list[str], **_) -> EvalPrediction:
        prompt_embeddings = self.embedder.encode(
            prompts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=True
        )
        distances = torch.cdist(prompt_embeddings, self.cluster_centers)
        weights = torch.softmax(-distances, dim=1)
        predictions = (weights[:, :, None] * self.cluster_scores).sum(dim=1)
        return EvalPrediction(predictions=predictions.cpu().numpy(), label_ids=None)
