import os
import numpy as np
import random
from typing import Any, Generator, Literal
from datasets import Dataset, concatenate_datasets

from args import DataArguments
from utils import load_json, get_logger
from utils.math import mean
from .readers import read_tag_scores, read_training_data, DATASET_READERS

logger = get_logger(__name__)


def load_train_datasets(data_args: DataArguments) -> tuple[Dataset, Dataset]:
    """Load and training and validation datasets.

    Args:
        data_args: Data arguments.

    Returns:
        The training and validation datasets.
    """
    # Load data
    assert data_args.train_datasets, "Training dataset is not provided."
    if data_args.tag_score_weight != 0:
        tag_scores = read_tag_scores(os.path.join(data_args.data_dir, "train", "tag_scores"), data_args.candidate_llms)
    else:
        tag_scores = {}
    train_datasets, eval_datasets = [], []
    for dataset_name in data_args.train_datasets:
        dataset_path = os.path.join(data_args.data_dir, "train", dataset_name)
        eval_ids = load_json(os.path.join(dataset_path, "../eval_ids.json"))
        gen_kwargs = {
            "data_dir": dataset_path,
            "candidate_llms": data_args.candidate_llms,
            "score_normalization": data_args.score_normalization,
            "eval_ids": eval_ids,
            "cluster_id_name": data_args.cluster_id_name,
            "tag_scores": tag_scores,
            "tag_score_weight": data_args.tag_score_weight,
        }
        train_dataset: Dataset = Dataset.from_generator(_generate_train_example, gen_kwargs=gen_kwargs)  # type: ignore
        eval_dataset: Dataset = Dataset.from_generator(_generate_eval_example, gen_kwargs=gen_kwargs)  # type: ignore

        # Slice training dataset
        train_dataset = train_dataset.shuffle()
        start = int(len(train_dataset) * data_args.train_slice[0])
        end = int(len(train_dataset) * data_args.train_slice[1])
        train_dataset = train_dataset.select(range(start, end))  # type: ignore

        logger.warning_rank0(
            f"Loaded {dataset_name} with {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples."
        )
        train_datasets.append(train_dataset)
        eval_datasets.append(eval_dataset)

    # Merge datasets
    train_dataset = concatenate_datasets(train_datasets).shuffle()
    eval_dataset = concatenate_datasets(eval_datasets)

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), data_args.max_train_samples))

    return train_dataset, eval_dataset


def load_test_datasets(data_args: DataArguments) -> Generator[tuple[str, Dataset], Any, None]:
    """Load test datasets.

    Args:
        data_args: Data arguments.

    Returns:
        The test datasets.
    """
    for dataset_name in data_args.test_datasets:
        dataset: Dataset = Dataset.from_generator(
            DATASET_READERS.get(dataset_name, read_training_data),
            gen_kwargs={
                "data_dir": os.path.join(data_args.data_dir, "test", dataset_name),
                "candidate_llms": data_args.candidate_llms,
            },
        )  # type: ignore
        yield dataset_name, dataset


def _generate_train_example(
    data_dir: str,
    candidate_llms: list[str],
    score_normalization: Literal["none", "min-max", "softmax"],
    eval_ids: set[str] = set(),
    cluster_id_name: str | None = None,
    tag_scores: dict[str, list[float]] = {},
    tag_score_weight: float = 0,
) -> Generator[dict[str, list[dict[str, str]]], Any, None]:
    for item in read_training_data(data_dir, candidate_llms, score_normalization=True, cluster_id_name=cluster_id_name):
        if item["prompt_id"] in eval_ids or any(r == "" for r in item["responses"]):
            continue
        item["scores"] = _enhance_with_tag_score(item["scores"], item["tags"], tag_scores, tag_score_weight)
        item["scores"] = _normalize_score(item["scores"], score_normalization)
        yield item


def _generate_eval_example(
    data_dir: str,
    candidate_llms: list[str],
    score_normalization: Literal["none", "min-max", "softmax"],
    eval_ids: set[str] = set(),
    cluster_id_name: str | None = None,
    tag_scores: dict[str, list[float]] = {},
    tag_score_weight: float = 0,
) -> Generator[dict[str, list[dict[str, str]]], Any, None]:
    for item in read_training_data(data_dir, candidate_llms, score_normalization=True, cluster_id_name=cluster_id_name):
        if item["prompt_id"] in eval_ids:
            item["scores"] = _enhance_with_tag_score(item["scores"], item["tags"], tag_scores, tag_score_weight)
            item["scores"] = _normalize_score(item["scores"], score_normalization)
            yield item


def _normalize_score(
    scores: list[float], normalization_method: Literal["none", "min-max", "softmax", "standardization"]
) -> list[float]:
    if normalization_method == "none":
        return scores
    elif normalization_method == "min-max":
        score_array = np.array(scores)
        min_score = score_array.min()
        max_score = score_array.max()
        score_array = (score_array - min_score) / ((max_score - min_score) if max_score != min_score else 1)
        return score_array.tolist()
    elif normalization_method == "softmax":
        score_array = np.array(scores)
        exp_score_array = np.exp(score_array)
        score_array = exp_score_array / exp_score_array.sum()
        return score_array.tolist()
    elif normalization_method == "standardization":
        score_array = np.array(scores)
        score_mean = score_array.mean()
        score_std = score_array.std()
        score_array = (score_array - score_mean) / (score_std if score_std != 0 else 1)
        return score_array.tolist()
    else:
        raise ValueError(f"Invalid normalization method: {normalization_method}.")


def _enhance_with_tag_score(
    scores: list[float], tags: list[str], tag_scores: dict[str, list[float]], tag_score_weight: float
) -> list[float]:
    if tag_score_weight != 0:
        for i in range(len(scores)):
            tag_score = mean([tag_scores[tag][i] for tag in tags])
            scores[i] = (1 - tag_score_weight) * scores[i] + tag_score_weight * tag_score
    return scores
