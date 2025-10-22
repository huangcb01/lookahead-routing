import os
import torch

from utils import save_jsonl


def write_ultrafeedback(data: list[dict], data_dir: str, candidate_llms: list[str], write_instructions: bool = False):
    if write_instructions:
        instructions = [
            {k: v for k, v in item.items() if k in ["prompt_id", "prompt"] or "cluster" in k} for item in data
        ]
        save_jsonl(instructions, os.path.join(data_dir, "..", "instructions.jsonl"))
    for i, llm in enumerate(candidate_llms):
        llm_data = [
            {
                "prompt_id": item["prompt_id"],
                "response": item["responses"][i],
                "score": item["scores"][i],
            }
            for item in data
        ]
        save_jsonl(llm_data, os.path.join(data_dir, f"{llm}.jsonl"))
    if "hidden_states" in data[0]:
        for i, llm in enumerate(candidate_llms):
            llm_data = [
                {
                    "prompt_id": item["prompt_id"],
                    "hidden_states": item["hidden_states"][i],
                }
                for item in data
            ]
            torch.save(llm_data, os.path.join(data_dir, f"hs_{llm}.pt"))
