import os
from typing import Any, Generator

import torch

from utils import argmax, mean, extract_variables, load_json, read_jsonl, load_json, get_logger


logger = get_logger(__name__)


def read_training_data(
    data_dir: str,
    candidate_llms: list[str],
    score_normalization: bool = False,
    read_hidden_states: bool = False,
    cluster_id_name: str | None = None,
) -> Generator[dict[str, Any], Any, None]:
    # RESPONSE_KEY = "prediction" if "SelfOSSInstructSC2" in data_dir else "response"
    RESPONSE_KEY = "response"
    n_candidates = len(candidate_llms)
    if score_normalization:
        max_score, min_score = None, None
        for item in read_training_data(data_dir, candidate_llms, score_normalization=False):
            scores = item["scores"]
            max_score = max(max_score, max(scores)) if max_score is not None else max(scores)
            min_score = min(min_score, min(scores)) if min_score is not None else min(scores)
        logger.info(f"Normalizing scores in the range [{min_score}, {max_score}]")
    instruction_iter = read_jsonl(os.path.join(data_dir, "..", "instructions.jsonl"))
    response_iters = [read_jsonl(os.path.join(data_dir, f"{llm}.jsonl")) for llm in candidate_llms]
    if read_hidden_states:
        hidden_states = [torch.load(os.path.join(data_dir, f"hs_{llm}.pt")) for llm in candidate_llms]
    if candidate_llms:
        for i, (instruction, responses) in enumerate(zip(instruction_iter, zip(*response_iters))):
            item = {
                "prompt_id": instruction["prompt_id"],
                "prompt": instruction["prompt"].strip(),
                "tags": instruction["tags"],
                "responses": [response[RESPONSE_KEY] for response in responses],
            }
            if "score" in responses[0]:
                item["scores"] = [
                    (
                        (response["score"] - min_score) / (max_score - min_score)
                        if score_normalization
                        else response["score"]
                    )
                    for response in responses
                ]
                item["best_model_idx"] = argmax(item["scores"])
            if read_hidden_states:
                item["hidden_states"] = [hidden_states[j][i]["hidden_states"] for j in range(n_candidates)]
            if cluster_id_name is not None:
                item["cluster_id"] = instruction[cluster_id_name]
            yield item
    else:
        for instruction in instruction_iter:
            item = {"prompt_id": instruction["prompt_id"], "prompt": instruction["instruction"].strip()}
            yield item


def read_predictions(data_dir: str) -> Generator[dict[str, Any], Any, None]:
    for item in load_json(data_dir):
        yield {**item, "best_model_idx": argmax(item["scores"])}


def read_mtbench(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    MT_BENCH_PROMPTS = {
        "single-v1": '[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{answer}\n[The End of Assistant\'s Answer]',
        "single-math-v1": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
        "single-v1-multi-turn": "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer}\n\n<|The End of Assistant A's Conversation with User|>",
        "single-math-v1-multi-turn": "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1_2}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2_2}\n\n### Assistant A:\n{answer}\n\n<|The End of Assistant A's Conversation with User|>",
    }
    questions = read_jsonl(os.path.join(data_dir, "question.jsonl"))
    judgments = [
        {
            f"{item['question_id']}_{item['turn']}": {
                "response": extract_variables(item["user_prompt"], MT_BENCH_PROMPTS[item["judge"][1]])["answer"],
                "score": item["score"],
            }
            for item in read_jsonl(os.path.join(data_dir, "judgments", f"gpt-4-0125-preview_judge_{llm}_single.jsonl"))
        }
        for llm in candidate_llms
    ]
    for question in questions:
        qid = question["question_id"]
        turn_scores = [(judgment[f"{qid}_1"]["score"], judgment[f"{qid}_2"]["score"]) for judgment in judgments]
        scores = [mean(s) for s in turn_scores]
        yield {
            "prompt_id": qid,
            "category": question["category"],
            "prompt": question["turns"][0].strip(),
            "responses": [judgment[f"{qid}_1"]["response"] for judgment in judgments],
            "scores": scores,
            "turn_scores": turn_scores,
            "best_model_idx": argmax(scores),
        }


def read_alpacaeval(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    instructions = read_jsonl(os.path.join(data_dir, "instructions.jsonl"))
    annotations = [
        [
            {"response": item["output_2"], "preference": item["preference"]}
            for item in load_json(os.path.join(data_dir, llm, "weighted_alpaca_eval_gpt4_turbo", "annotations.json"))
        ]
        for llm in candidate_llms
    ]
    for i, instruction in enumerate(instructions):
        scores = [annotations[j][i]["preference"] for j in range(len(candidate_llms))]
        yield {
            "prompt": instruction["instruction"].strip(),
            "reference": instruction["reference"],
            "responses": [annotations[j][i]["response"] for j in range(len(candidate_llms))],
            "scores": scores,
            "best_model_idx": argmax(scores),
        }


def read_arenahard(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    BASELINE_MODEL = "gpt-4-0314"
    MULTIPLIER = 3

    def get_battles(games: list[dict[str, str]]):
        results = []

        output = {"model_a": BASELINE_MODEL, "model_b": "router"}
        game = games[0]
        weight = 1
        if game["score"] == "A=B":
            output["winner"] = "tie"
        elif game["score"] == "A>B":
            output["winner"] = "model_a"
        elif game["score"] == "A>>B" or game["score"] is None:
            output["winner"] = "model_a"
            weight = MULTIPLIER
        elif game["score"] == "B>A":
            output["winner"] = "model_b"
        elif game["score"] == "B>>A":
            output["winner"] = "model_b"
            weight = MULTIPLIER
        else:
            weight = 0
        results += [output] * weight

        output = {"model_a": BASELINE_MODEL, "model_b": "router"}
        game = games[1]
        weight = 1
        if game["score"] == "A=B":
            output["winner"] = "tie"
        elif game["score"] == "A>B":
            output["winner"] = "model_b"
        elif game["score"] == "A>>B":
            output["winner"] = "model_b"
            weight = MULTIPLIER
        elif game["score"] == "B>A":
            output["winner"] = "model_a"
        elif game["score"] == "B>>A" or game["score"] is None:
            output["winner"] = "model_a"
            weight = MULTIPLIER
        else:
            weight = 0
        results += [output] * weight
        return results

    def compute_score(games: list[dict[str, str]]):
        score_map = {
            "A>>B": 3,
            "A>B": 1,
            "A=B": 0,
            "B>A": -1,
            "B>>A": -3,
        }
        if games[0]["score"] is None or games[1]["score"] is None:
            return -6
        else:
            return score_map[games[1]["score"]] - score_map[games[0]["score"]]

    questions = read_jsonl(os.path.join(data_dir, "question.jsonl"))
    responses = [
        {
            item["question_id"]: item["choices"][0]["turns"][0]["content"]
            for item in read_jsonl(os.path.join(data_dir, "model_answer", f"{llm}.jsonl"))
        }
        for llm in candidate_llms
    ]
    token_lens = [
        {
            item["question_id"]: (
                item["choices"][0]["turns"][0]["token_len"]
                if "token_len" in item["choices"][0]["turns"][0]
                else item["conv_metadata"]["token_len"]
            )
            for item in read_jsonl(os.path.join(data_dir, "model_answer", f"{llm}.jsonl"))
        }
        for llm in candidate_llms
    ]
    battles = [
        {
            item["question_id"]: get_battles(item["games"])
            for item in read_jsonl(os.path.join(data_dir, "model_judgment", f"{llm}.jsonl"))
        }
        for llm in candidate_llms
    ]
    scores = [
        {
            item["question_id"]: compute_score(item["games"])
            for item in read_jsonl(os.path.join(data_dir, "model_judgment", f"{llm}.jsonl"))
        }
        for llm in candidate_llms
    ]
    for question in questions:
        try:
            pid = question["question_id"]
            s = [score[pid] for score in scores]
            yield {
                "prompt_id": pid,
                "prompt": question["turns"][0]["content"].strip(),
                "responses": [response[pid] for response in responses],
                "scores": s,
                "battles": [battle[pid] for battle in battles],
                "token_lens": [token_len[pid] for token_len in token_lens],
                "best_model_idx": argmax(s),
            }
        except KeyError:
            logger.warning_rank0(f"ArenaHard KeyError: {question['question_id']}")
            continue


def read_mmlupro(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    subsets = [file_name[: -len(".json")] for file_name in os.listdir(os.path.join(data_dir, candidate_llms[0]))]
    for subset in subsets:
        subset_iterators: list[dict[str, dict[str, Any]]] = [
            load_json(os.path.join(data_dir, llm, f"{subset}.json"))["details"] for llm in candidate_llms  # type: ignore
        ]
        first_iterator = subset_iterators[0]
        for prompt_id, item in first_iterator.items():
            prompt = item["prompt"][0]["prompt"]
            assert all(iterator[prompt_id]["prompt"][0]["prompt"] == prompt for iterator in subset_iterators[1:])
            responses = [iterator[prompt_id]["origin_prediction"] for iterator in subset_iterators]
            scores = [float(iterator[prompt_id]["predictions"] == item["references"]) for iterator in subset_iterators]
            yield {
                "prompt_id": f"{subset}_{prompt_id}",
                "prompt": prompt.strip(),
                "responses": responses,
                "scores": scores,
                "best_model_idx": argmax(scores),
                "subset": subset,
            }


def read_opencompass(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    iterators: list[dict[str, dict[str, Any]]] = [
        load_json(os.path.join(data_dir, f"{llm}.json"))["details"] for llm in candidate_llms  # type: ignore
    ]
    first_iterator = iterators[0]
    for prompt_id, item in first_iterator.items():
        prompt = item["prompt"][0]["prompt"]
        assert all(iterator[prompt_id]["prompt"][0]["prompt"] == prompt for iterator in iterators[1:])
        responses = [iterator[prompt_id]["origin_prediction"] for iterator in iterators]
        scores = [float(iterator[prompt_id]["correct"]) for iterator in iterators]
        yield {
            "prompt_id": prompt_id,
            "prompt": prompt.strip(),
            "responses": responses,
            "scores": scores,
            "best_model_idx": argmax(scores),
        }


def read_code(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    predictions: list[dict[str, dict[str, Any]]] = [
        load_json(os.path.join(data_dir, "predictions", f"{llm}.json")) for llm in candidate_llms  # type: ignore
    ]
    results: list[dict[str, dict[str, Any]]] = [
        load_json(os.path.join(data_dir, "results", f"{llm}.json"))["details"] for llm in candidate_llms  # type: ignore
    ]
    first_predictions = predictions[0]
    for prompt_id, item in first_predictions.items():
        prompt = item["origin_prompt"][0]["prompt"]
        assert all(iterator[prompt_id]["origin_prompt"][0]["prompt"] == prompt for iterator in predictions[1:])
        responses = [iterator[prompt_id]["prediction"] for iterator in predictions]
        scores = [float(iterator[prompt_id]["is_correct"]) for iterator in results]
        yield {
            "prompt_id": prompt_id,
            "prompt": prompt.strip(),
            "responses": responses,
            "scores": scores,
            "best_model_idx": argmax(scores),
        }


def read_humaneval(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    predictions: list[dict[str, dict[str, Any]]] = [
        load_json(os.path.join(data_dir, "predictions", f"{llm}.json")) for llm in candidate_llms  # type: ignore
    ]
    results: list[dict[str, dict[str, Any]]] = [
        load_json(os.path.join(data_dir, "results", f"{llm}.json"))["details"] for llm in candidate_llms  # type: ignore
    ]
    first_predictions = predictions[0]
    for prompt_id, item in first_predictions.items():
        prompt = item["origin_prompt"][0]["prompt"]
        assert all(iterator[prompt_id]["origin_prompt"][0]["prompt"] == prompt for iterator in predictions[1:])
        responses = [iterator[prompt_id]["completion"] for iterator in results]
        scores = [float(iterator[prompt_id]["is_correct"]) for iterator in results]
        yield {
            "prompt_id": prompt_id,
            "prompt": prompt.strip(),
            "responses": responses,
            "scores": scores,
            "best_model_idx": argmax(scores),
        }


def read_mbpp(data_dir: str, candidate_llms: list[str]) -> Generator[dict[str, Any], Any, None]:
    predictions: list[dict[str, dict[str, Any]]] = [
        load_json(os.path.join(data_dir, "predictions", f"{llm}.json")) for llm in candidate_llms  # type: ignore
    ]
    results: list[dict[str, dict[str, Any]]] = [
        load_json(os.path.join(data_dir, "results", f"{llm}.json"))["details"] for llm in candidate_llms  # type: ignore
    ]
    first_predictions = predictions[0]
    for prompt_id, item in first_predictions.items():
        prompt = item["origin_prompt"][0]["prompt"]
        assert all(iterator[prompt_id]["origin_prompt"][0]["prompt"] == prompt for iterator in predictions[1:])
        responses = [iterator[prompt_id]["programs"] for iterator in results]
        scores = [float(iterator[prompt_id]["is_correct"]) for iterator in results]
        yield {
            "prompt_id": prompt_id,
            "prompt": prompt.strip(),
            "responses": responses,
            "scores": scores,
            "best_model_idx": argmax(scores),
        }


def read_tag_scores(data_dir: str, candidate_llms: list[str]) -> dict[str, list[float]]:
    data: list[dict[str, float]] = [load_json(os.path.join(data_dir, f"{llm}.json")) for llm in candidate_llms]  # type: ignore
    tag_scores = {k: [data[i][k] for i in range(len(candidate_llms))] for k in data[0].keys()}
    return tag_scores


DATASET_READERS = {
    "UltraFeedback": read_training_data,
    "MT-Bench": read_mtbench,
    "AlpacaEval": read_alpacaeval,
    "ArenaHard": read_arenahard,
    "MMLU-Pro": read_mmlupro,
    "GPQA-Diamond": read_opencompass,
    "GSM8k": read_opencompass,
    "MATH": read_opencompass,
    "HumanEval": read_code,
    "MBPP": read_code,
}
