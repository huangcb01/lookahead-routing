import os
import fire
from typing import Literal
from dataclasses import asdict

from args import DataArguments
from data.dataset import load_test_datasets
from routers import ROUTERS
from data.readers import DATASET_READERS, read_training_data
from metrics import compute_metrics
from utils import json2str, load_json, save_json


def evaluate(
    router_type: Literal["random", "oracle", "fixed", "reward", "knn"],
    candidate_llms: list[str],
    data_dir: str,
    test_datasets: list[str],
    output_dir: str,
    batch_size: int = 1,
    **router_kwargs,
):
    # Load model
    router_kwargs = {**router_kwargs, "candidate_llms": candidate_llms, "data_dir": data_dir, "batch_size": batch_size}
    router = ROUTERS[router_type](len(candidate_llms), **router_kwargs)

    # Load data
    data_args = DataArguments(data_dir=data_dir, test_datasets=test_datasets, candidate_llms=candidate_llms)
    datasets = load_test_datasets(data_args)

    # Inference
    metrics = {}
    for name, dataset in datasets:
        print(f"Evaluating {name}...")
        # Inference
        predictions = router.route(
            [item["prompt"] for item in dataset],
            responses=[item["responses"] for item in dataset],
            scores=[item["scores"] for item in dataset],
            batch_size=batch_size,
        )
        # Compute metrics and save results
        metrics[name] = compute_metrics(predictions, dataset, name)  # type: ignore
        results = metrics[name].pop("details")
        print(json2str(metrics[name]))
        save_json(results, os.path.join(output_dir, f"{name}_predictions.json"))

    # Print and save metrics
    print(json2str(metrics))
    save_json(metrics, os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    fire.Fire(evaluate)
