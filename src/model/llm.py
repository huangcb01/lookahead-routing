import json
import os
import requests
import contextlib
import gc
import torch
import openai
import multiprocessing as mp
from time import sleep
from typing import Any, Callable, Type
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm, trange
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.utils import is_torch_bf16_gpu_available
from openai.types import CompletionUsage

from utils import load_obj, save_obj


def _parse_output(outputs: RequestOutput | GenerateDecoderOnlyOutput) -> list[str]:
    """Default output parser that gets the response text from the output."""
    if isinstance(outputs, RequestOutput):  # vLLM output
        return [output.text for output in outputs.outputs]
    elif isinstance(outputs, GenerateDecoderOnlyOutput):  # Hugging Face output
        raise NotImplementedError("Implement output parser for GenerateDecoderOnlyOutput.")


class BaseLLM:
    """Base class for large language models."""

    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize the model.

        Args:
            model_name_or_path: The name or path of the model.
            **kwargs: Additional arguments to initialize the backend model.
        """
        self.tokenizer: PreTrainedTokenizerBase  # Defined in subclass.
        self.model_path = model_name_or_path
        self.lora_path = kwargs.get("lora_path")
        self.usage = {
            "completions": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def chat(
        self,
        messages: list[list[dict[str, str]]],
        sampling_params: SamplingParams,
        chunk_size: int = -1,
        progress_bar: bool = True,
        output_parser: Callable[[RequestOutput], Any] = _parse_output,
        cache_dir: str | None = None,
        **kwargs,
    ) -> list[Any]:
        """Generate responses for a list of conversations.

        Args:
            messages: A list of completions, each of which is a list of messages and each message is a dictionary with "role" and "content" keys.
            sampling_params: Sampling parameters.
            chunk_size: The cache will be saved to disk after each chunk.
            progress_bar: Whether to show the progress bar.
            cache_dir: The directory to cache the responses.

        Returns:
            A list of outputs of the `output_parser` function for each conversation.
            When using default `output_parser`, each element of the returned list is a list of candidate responses.
        """
        if chunk_size == -1:
            chunk_size = len(messages)
        results = []
        n_chunks = (len(messages) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            print(f"Generating chunk {i+1}/{n_chunks}...")
            if cache_dir and os.path.exists(os.path.join(cache_dir, f"chunk_{i}.pickle")):
                print("Cached responses found, skip generating.")
                continue
            chunk_messages = messages[i * chunk_size : (i + 1) * chunk_size]
            chunk_outputs = self._chat(chunk_messages, sampling_params, progress_bar, **kwargs)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with mp.Pool(processes=min(os.cpu_count() or 1, chunk_size)) as pool:
                chunk_results = pool.map(output_parser, chunk_outputs)
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            if cache_dir:
                save_obj(chunk_results, os.path.join(cache_dir, f"chunk_{i}.pickle"))
            else:
                results += chunk_results
        if cache_dir:
            print("Merging cached responses...")
            for i in range(n_chunks):
                chunk_results = load_obj(os.path.join(cache_dir, f"chunk_{i}.pickle"))
                results += chunk_results
        return results

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        chunk_size: int = -1,
        progress_bar: bool = True,
        output_parser: Callable[[RequestOutput], Any] = _parse_output,
        cache_dir: str | None = None,
    ) -> list[Any]:
        """Generate responses for a list of prompts.

        Args:
            prompts: A list of prompts.
            sampling_params: Sampling parameters.
            chunk_size: The cache will be saved to disk after each chunk.
            progress_bar: Whether to show the progress bar.
            cache_dir: The directory to cache the responses.

        Returns:
            A list of outputs of the `output_parser` function for each conversation.
            When using default `output_parser`, each element of the returned list is a list of candidate responses.
        """
        if chunk_size == -1:
            chunk_size = len(prompts)
        results = []
        n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            print(f"Generating chunk {i+1}/{n_chunks}...")
            if cache_dir and os.path.exists(os.path.join(cache_dir, f"chunk_{i}.pickle")):
                print("Cached responses found, skip generating.")
                continue
            chunk_prompts = prompts[i * chunk_size : (i + 1) * chunk_size]
            chunk_outputs = self._generate(chunk_prompts, sampling_params, progress_bar)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with mp.Pool(processes=min(os.cpu_count() or 1, chunk_size)) as pool:
                chunk_results = pool.map(output_parser, chunk_outputs)
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            if cache_dir:
                save_obj(chunk_results, os.path.join(cache_dir, f"chunk_{i}.pickle"))
            else:
                results += chunk_results
        if cache_dir:
            print("Merging cached responses...")
            for i in range(n_chunks):
                chunk_results = load_obj(os.path.join(cache_dir, f"chunk_{i}.pickle"))
                results += chunk_results
        return results

    def _chat(
        self, messages: list[list[dict[str, str]]], sampling_params: SamplingParams, progress_bar: bool = True
    ) -> list[RequestOutput]:
        raise NotImplementedError("Implement chat() method in subclass.")

    def _generate(
        self, prompts: list[str], sampling_params: SamplingParams, progress_bar: bool = True
    ) -> list[RequestOutput]:
        raise NotImplementedError("Implement chat() method in subclass.")


class VLLM(BaseLLM):
    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize the vLLM model.

        Args:
            model_name_or_path: The name or path of the model.
            **kwargs: Additional arguments to initialize the `vllm.LLM` object.
        """
        super().__init__(model_name_or_path, **kwargs)
        llm_kwargs = {
            "trust_remote_code": True,
            "dtype": "bfloat16" if is_torch_bf16_gpu_available() else "float16",
            "tensor_parallel_size": torch.cuda.device_count(),
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "num_scheduler_steps": 16,
            "swap_space": 32 // torch.cuda.device_count(),
            "gpu_memory_utilization": 0.95,
            "preemption_mode": "swap",
            **kwargs,
        }
        self.llm = LLM(
            model=model_name_or_path,
            **llm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()  # type: ignore

    def __del__(self):
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()  # type: ignore
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _chat(
        self, messages: list[list[dict[str, str]]], sampling_params: SamplingParams, progress_bar: bool = True, **kwargs
    ) -> list[RequestOutput]:
        if "system" not in self.tokenizer.chat_template:  # type: ignore
            print("Warning: System message not supported, concatenating it with the first user message.")
            for message in messages:
                if message[0]["role"] == "system":
                    message[1]["content"] = message[0]["content"] + "\n" + message[1]["content"]
                    message.pop(0)
        return self.llm.chat(messages, sampling_params, use_tqdm=progress_bar, **kwargs)  # type: ignore

    def _generate(
        self, prompts: list[str], sampling_params: SamplingParams, progress_bar: bool = True
    ) -> list[RequestOutput]:
        return self.llm.generate(prompts, sampling_params, use_tqdm=progress_bar)  # type: ignore


class HFLLM(BaseLLM):
    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize the Hugging Face model.

        Args:
            model_name_or_path: The name or path of the model.
            **kwargs: Additional arguments passed to the `AutoModelForCausalLM.from_pretrained` method.
        """
        super().__init__(model_name_or_path, **kwargs)
        self.batch_size = kwargs.pop("batch_size", 1)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            **kwargs,
        )
        if self.lora_path:
            self.llm.load_adapter(self.lora_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __del__(self):
        super().__del__()
        del self.llm
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sleep(4)

    def _chat(
        self, messages: list[list[dict[str, str]]], sampling_params: SamplingParams, progress_bar: bool = True
    ) -> list[RequestOutput]:
        if "system" not in self.tokenizer.chat_template:
            print("Warning: System message not supported, concatenating it with the first user message.")
            for message in messages:
                if message[0]["role"] == "system":
                    message[1]["content"] = message[0]["content"] + "\n" + message[1]["content"]
                    message.pop(0)
        responses = []
        for i in trange(0, len(messages), self.batch_size, disable=not progress_bar):
            batch_messages = messages[i : i + self.batch_size]
            inputs = self.tokenizer.apply_chat_template(
                batch_messages, add_generation_prompt=True, return_tensors="pt", padding=True, return_dict=True
            ).to(  # type: ignore
                self.llm.device
            )
            input_length = inputs.input_ids.shape[1]
            generated_ids = self.llm.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=sampling_params.n,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_new_tokens=sampling_params.max_tokens,
            ).reshape(len(batch_messages), sampling_params.n, -1)
            responses += [
                [
                    response.strip()
                    for response in self.tokenizer.batch_decode(ids[:, input_length:], skip_special_tokens=True)
                ]
                for ids in generated_ids
            ]
        return responses


class OpenAI(BaseLLM):
    """Use `openai` library"""

    def __init__(
        self,
        model_name: str,
        num_proc: int = 1,
        num_retries: int = 5,
        retry_wait: int = 2,
        **kwargs,
    ):
        """Initialize the OpenAI model.

        Args:
            model_name: The name of the model.
            cache_dir: The directory to cache the responses.
            num_proc: The number of processes to use for parallel requests.
            num_retries: The number of retries for each request.
            retry_wait: The wait time between retries.
            **kwargs: Additional arguments to initialize the `openai.OpenAI` object.
        """
        super().__init__(model_name)
        self.num_proc = num_proc
        self.num_retries = num_retries
        self.retry_wait = retry_wait
        self.client = openai.OpenAI(**kwargs)

    def _chat(
        self, messages: list[list[dict[str, str]]], sampling_params: SamplingParams, progress_bar: bool = True
    ) -> list[RequestOutput]:
        with ThreadPoolExecutor(max_workers=self.num_proc) as p:
            n_completions = len(messages)
            responses = list(
                tqdm(
                    p.map(self._request, messages, [sampling_params] * n_completions),
                    total=n_completions,
                    disable=not progress_bar,
                )
            )
        return responses

    def _request(self, messages: list[dict[str, str]], sampling_params: SamplingParams) -> RequestOutput:
        for i in range(self.num_retries):
            try:
                raw_response = self.client.chat.completions.create(
                    messages=messages,  # type: ignore
                    model=self.model_path,
                    max_completion_tokens=sampling_params.max_tokens,
                    n=sampling_params.n,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                )
                if raw_response.usage:
                    self._update_usage(raw_response.usage)
                output = RequestOutput(
                    request_id=raw_response.id,
                    prompt=None,
                    prompt_token_ids=None,
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(choice.index, choice.message.content.strip(), [], None, None)
                        for choice in raw_response.choices
                    ],
                    finished=True,
                )
                return output
            except Exception as e:
                print(f"Retry {i+1}/{self.num_retries}: Error {e} for message {messages}")
            sleep(self.retry_wait)
        raise RuntimeError(f"Failed to get response after {self.num_retries} retries.")

    def _update_usage(self, usage: CompletionUsage):
        self.usage["completions"] += 1
        self.usage["completion_tokens"] += usage.completion_tokens
        self.usage["prompt_tokens"] += usage.prompt_tokens
        self.usage["total_tokens"] += usage.total_tokens


# TODO: Fix
class RequestLLM(OpenAI):
    def __init__(self, model_path: str, cache_dir: str | None = None, **kwargs):
        BaseLLM.__init__(self, model_path, cache_dir, **kwargs)
        self.base_url = kwargs.pop("base_url", "")
        self.api_key = kwargs.pop("api_key", "")
        self.num_proc = kwargs.pop("num_proc", 1)
        self.num_retries = kwargs.pop("num_retries", 5)

    def _request(self, messages: list[dict[str, str]], sampling_params: SamplingParams) -> list[str]:
        for i in range(self.num_retries):
            try:
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "messages": messages,
                    "model": self.model_path,
                    "max_tokens": sampling_params.max_tokens,
                    "n": sampling_params.n,
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                }
                payload = json.dumps(payload)
                if payload in self.cache:
                    raw_response = self.cache[payload]
                else:
                    raw_response = requests.request("POST", self.base_url, headers=headers, data=payload, timeout=120)
                    raw_response = raw_response.content.decode("utf-8")
            except Exception as e:
                print(f"Retry {i + 1} / {self.num_retries}: Request error {e} for message {messages}")
                continue
            try:
                raw_response = json.loads(raw_response)["data"]["response"]
                self._update_usage(raw_response["usage"])
                responses = [choice["message"]["content"] for choice in raw_response["choices"]]
                self.cache[payload] = responses
                return responses
            except Exception as e:
                print(f"Retry {i + 1} / {self.num_retries}: Parsing error {e} for response {raw_response}")
            sleep(self.retry_wait)
        return []

    def _update_usage(self, usage: dict):
        self.usage["completions"] += 1
        self.usage["completion_tokens"] += usage["completion_tokens"]
        self.usage["prompt_tokens"] += usage["prompt_tokens"]
        self.usage["total_tokens"] += usage["total_tokens"]


LLMS: dict[str, Type[BaseLLM]] = {
    "vllm": VLLM,
    "huggingface": HFLLM,
    "openai": OpenAI,
    "request": RequestLLM,
}
