import fire
from typing import Literal

from model.llm import LLMS, SamplingParams
from utils import read_jsonl, save_jsonl


def generate_responses(
    model_name_or_path: str,
    data_path: str,
    output_path: str,
    cache_dir: str | None = None,
    engine: Literal["huggingface", "vllm"] = "huggingface",
    chunk_size: int = -1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 4096,
    **engine_kwargs,
):
    # Load model and data
    llm = LLMS[engine](model_name_or_path, cache_dir, **engine_kwargs)
    data = list(read_jsonl(data_path))

    # Generate responses
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    messages = [[{"role": "user", "content": item["instruction"]}] for item in data]
    responses = llm.chat(messages, sampling_params, chunk_size=chunk_size)
    results = [
        {
            "prompt_id": item["prompt_id"],
            "response": response[0],
            "response_toknes": len(llm.tokenizer.encode(response[0], add_special_tokens=False)),
        }
        for item, response in zip(data, responses)
    ]

    # Save responses
    save_jsonl(results, output_path)


if __name__ == "__main__":
    fire.Fire(generate_responses)
