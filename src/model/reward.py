import torch
from typing import Literal
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import is_torch_bf16_gpu_available, is_flash_attn_2_available
from vllm import LLM
from tqdm import tqdm


class RewardModel:
    def __init__(
        self,
        model_name_or_path: str,
        engine: Literal["huggingface", "vllm"] = "huggingface",
        **kwargs,
    ):
        """Initialize the reward model.

        Args:
            model_name_or_path: The Hugging Face model name or local path of the model.
            engine: The engine to use for computing the reward.
            kwargs: Additional keyword arguments to initialize the model.
        """
        self.engine = engine
        if self.engine == "huggingface":
            init_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": "bfloat16" if is_torch_bf16_gpu_available() else "float16",
                "attn_implementation": "flash_attention_2" if is_flash_attn_2_available() else None,
                **kwargs,
            }
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **init_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        elif self.engine == "vllm":
            init_kwargs = {
                "trust_remote_code": True,
                "dtype": "bfloat16" if is_torch_bf16_gpu_available() else "float16",
                "tensor_parallel_size": torch.cuda.device_count(),
                "gpu_memory_utilization": 0.95,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "num_scheduler_steps": 16,
                "preemption_mode": "swap",
                "swap_space": 32 // torch.cuda.device_count(),
                **kwargs,
            }
            self.model = LLM(model_name_or_path, task="reward", **init_kwargs)
            self.tokenizer = self.model.get_tokenizer()
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def compute_reward(self, messages: list[list[dict[str, str]]], output_hidden_states: bool = False):
        """Compute the reward score for each prompt-response pair."""
        if self.engine == "huggingface":
            rewards, hidden_states = [], []
            for message in tqdm(messages):
                input_ids = self.tokenizer.apply_chat_template(message, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=output_hidden_states)  # type: ignore
                    rewards.append(outputs.logits.squeeze().item())
                    if output_hidden_states:
                        hidden_states.append(outputs.hidden_states[-1][0, -1, :].cpu())
            if output_hidden_states:
                return torch.tensor(rewards), hidden_states
            else:
                return torch.tensor(rewards)
        elif self.engine == "vllm":
            prompts: list[str] = [self.tokenizer.apply_chat_template(message, tokenize=False) for message in messages]  # type: ignore
            outputs = self.model.encode(prompts)
            print(outputs[0].outputs.data.shape)
            exit()
            raise NotImplementedError("VLLM engine is not implemented yet.")
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
