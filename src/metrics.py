import pandas
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerBase
from datasets import Dataset
from alpaca_eval.metrics import get_length_controlled_winrate

from utils import mean, compute_mle_elo, predict_win_rate


def compute_metrics(
    eval_pred: EvalPrediction,
    test_set: Dataset,
    dataset_name: str = "UltraFeedback",
    tokenizer: PreTrainedTokenizerBase | None = None,
):
    """Compute metrics.

    Args:
        eval_pred: The evaluation prediction.
        test_set: The test set.
        dataset_name: The name of the dataset.
        tokenizer: The tokenizer.

    Returns:
        A dictionary of metrics.
    """
    if isinstance(eval_pred.predictions, tuple):
        routing_logits, output_token_ids = eval_pred.predictions  # type: ignore
        assert tokenizer is not None, "Tokenizer must be provided to decode responses."
        responses = [
            tokenizer.batch_decode(output_token_ids[i], skip_special_tokens=False)
            for i in range(output_token_ids.shape[0])
        ]
    else:
        routing_logits = eval_pred.predictions
        responses = None
    predicted_indices = np.argmax(routing_logits, axis=1)
    details = [
        {
            **item,
            "logits": logits.tolist(),
            "target_idx": predicted_idx.item(),
            "correct": predicted_idx.item() == item["best_model_idx"],
            "score": item["scores"][predicted_idx],
        }
        for item, logits, predicted_idx in zip(test_set, routing_logits, predicted_indices)
    ]
    if responses is not None:
        for item, response in zip(details, responses):
            item["predicted_responses"] = response
    n_candidates = len(test_set[0]["scores"])
    metrics: dict[str, int | float | list] = {
        "details": details,
        "accuracy": mean([p["correct"] for p in details]),
        "percentage": [mean([p["target_idx"] == i for p in details]) for i in range(n_candidates)],
    }
    DATASET_METRIC_FN = {
        "UltraFeedback": _compute_ultrafeedback_metrics,
        "MT-Bench": _compute_mtbench_metrics,
        "AlpacaEval": _compute_alpacaeval_metrics,
        "ArenaHard": _compute_arenahard_metrics,
        "MMLU-Pro": _compute_mmlupro_metrics,
        "GPQA-Diamond": _compute_accuracy_metrics,
        "GSM8k": _compute_accuracy_metrics,
        "MATH": _compute_accuracy_metrics,
        "HumanEval": _compute_accuracy_metrics,
        "MBPP": _compute_accuracy_metrics,
    }
    metric_fn = DATASET_METRIC_FN.get(dataset_name, _compute_ultrafeedback_metrics)
    metrics.update(metric_fn(details))
    return metrics


def _compute_accuracy_metrics(predictions: list[dict]) -> dict[str, float]:
    return {"task_accuracy": mean([pred["score"] for pred in predictions]) * 100}


def _compute_ultrafeedback_metrics(predictions: list[dict]) -> dict[str, float]:
    best_reward = mean([max(item["scores"]) for item in predictions])
    worst_reward = mean([min(item["scores"]) for item in predictions])
    normalize_reward = lambda reward: (reward - worst_reward) / (best_reward - worst_reward)
    for prediction in predictions:
        prediction["normalized_reward"] = normalize_reward(prediction["score"])
    return {
        "normalized_reward": mean([prediction["normalized_reward"] for prediction in predictions]),
        "reward": mean([prediction["score"] for prediction in predictions]),
    }


def _compute_mtbench_metrics(predictions: list[dict]) -> dict[str, float]:
    CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
    category_scores = {category: [] for category in CATEGORIES}
    turn_scores = ([], [])
    for prediction in predictions:
        category_scores[prediction["category"]].append(prediction["score"])
        turn_score = prediction["turn_scores"][prediction["target_idx"]]
        turn_scores[0].append(turn_score[0])
        turn_scores[1].append(turn_score[1])
    turn_scores = [mean(x) for x in turn_scores]
    metrics = {
        "score": mean(turn_scores),
        "turn_1": turn_scores[0],
        "turn_2": turn_scores[1],
        "categories": {category: mean(scores) for category, scores in category_scores.items()},
    }
    return metrics


def _compute_alpacaeval_metrics(predictions: list[dict]) -> dict[str, float]:
    predictions = [
        {
            "instruction": prediction["prompt"],
            "output_1": prediction["reference"],
            "generator_1": "gpt4_1106_preview",
            "output_2": prediction["responses"][prediction["target_idx"]],
            "generator_2": "router",
            "annotator": "weighted_alpaca_eval_gpt4_turbo",
            "preference": prediction["score"],
        }
        for prediction in predictions
    ]
    result_df = pandas.DataFrame(predictions)
    metrics = get_length_controlled_winrate(result_df)
    return {
        k: float(v)
        for k, v in metrics.items()
        if k in {"win_rate", "standard_error", "length_controlled_winrate", "lc_standard_error"}
    }


def _compute_arenahard_metrics(predictions: list[dict]) -> dict[str, float]:
    BASELINE_MODEL = "gpt-4-0314"
    battles = {"model_a": [], "model_b": [], "winner": []}
    for prediction in predictions:
        for k in battles:
            battles[k] += [battle[k] for battle in prediction["battles"][prediction["target_idx"]]]
    battles = pandas.DataFrame(battles)
    bt_model_coef = compute_mle_elo(battles, baseline_model=BASELINE_MODEL)
    win_rate = predict_win_rate(bt_model_coef.to_dict())
    win_rate = win_rate[BASELINE_MODEL].fillna(0.5).apply(lambda x: round(x * 100, 2))
    model_names = [name for name in bt_model_coef.index if name != BASELINE_MODEL]
    assert len(model_names) == 1
    return {"score": win_rate[model_names[0]]}


def _compute_mmlupro_metrics(predictions: list[dict]) -> dict[str, float | dict[str, float]]:
    subsets = set(item["subset"] for item in predictions)
    subset_correct = {subset: [] for subset in subsets}
    for pred in predictions:
        subset_correct[pred["subset"]].append(pred["score"])
    subset_accuracy = {k: mean(v) * 100 for k, v in subset_correct.items()}
    return {
        "micro_accuracy": mean([pred["score"] for pred in predictions]) * 100,
        "macro_accuracy": mean(list(subset_accuracy.values())),
        "subset_accuracy": subset_accuracy,
    }
