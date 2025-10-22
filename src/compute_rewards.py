import os
import fire
from typing import Literal
from itertools import islice, batched

from data.readers import read_training_data
from data.writers import write_ultrafeedback
from model.reward import RewardModel


def compute_reward(
    data_dir: str,
    model_name_or_path: str,
    engine: Literal["huggingface", "vllm"] = "huggingface",
    llms: list[str] = [],
    chunk_size: int = -1,
    **engine_kwargs,
):
    # Load model and data
    model = RewardModel(model_name_or_path, engine, **engine_kwargs)
    data = read_training_data(os.path.join(data_dir, "responses"), llms)
    output_dir = os.path.join(data_dir, model_name_or_path.split("/")[-1])
    try:
        results = list(read_training_data(output_dir, llms))
    except FileNotFoundError as e:
        results = []
    data = islice(data, len(results), None)  # Skip already computed data
    print(f"{len(results)} precomputed results loaded.")
    n_llms = len(llms)

    # Compute reward
    for i, chunk in enumerate(batched(data, chunk_size)):
        print(f"Computing reward for chunk {i+1}")
        messages = [
            [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["responses"][j]}]
            for item in chunk
            for j in range(n_llms)
        ]
        rewards, hidden_states = model.compute_reward(messages, output_hidden_states=True)
        for item, reward, hidden_state in zip(chunk, rewards.view(-1, n_llms), batched(hidden_states, n_llms)):
            results.append({**item, "scores": reward.tolist(), "hidden_states": list(hidden_state)})
        write_ultrafeedback(results, output_dir, llms)

if __name__ == "__main__":
    fire.Fire(compute_reward)
