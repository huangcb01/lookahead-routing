import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Any
from dataclasses import dataclass, asdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import PredictionOutput
from transformers.trainer_pt_utils import nested_detach
from datasets import Dataset
from typing_extensions import override
from trl.trainer.utils import pad

from args import CausalArguments
from utils import plot_trainer_state, get_logger
from model.causal import load_model_with_reward_head
from .utils import compute_routing_loss
from .base import BaseRouterTrainer


logger = get_logger(__name__)


@dataclass
class CausalCollator(DataCollatorMixin):
    pad_token_id: int
    model_token_ids: list[int]
    max_length: int | None = None
    use_logits_to_keep: bool = True
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of examples into a batch of tensors.

        Args:
            examples: List of examples.

        Returns:
            A dictionary of 4 items: input_ids, attention_mask, model_token_indices and rewards.
        """
        # Concatenate prompt, model ID and response
        input_ids = [
            torch.tensor(example["prompt_input_ids"] + [self.model_token_ids[i]] + response_input_ids)
            for example in examples
            for i, response_input_ids in enumerate(example["response_input_ids"])
        ]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        labels = [
            torch.tensor([-100] * (len(example["prompt_input_ids"]) + 1) + response_input_ids)
            for example in examples
            for response_input_ids in example["response_input_ids"]
        ]

        # Pad
        output = {
            "input_ids": pad(input_ids, padding_value=self.pad_token_id),  # shape=(batch_size*n_candidates, seq_length)
            "attention_mask": pad(attention_mask, padding_value=0),  # shape=(batch_size*n_candidates, seq_length)
            "labels": pad(labels, padding_value=-100),  # shape=(batch_size*n_candidates, seq_length)
            "model_token_indices": torch.tensor(
                [len(example["prompt_input_ids"]) for example in examples for _ in example["response_input_ids"]]
            ),  # shape=(batch_size*n_candidates,)
            "scores": torch.tensor([example["scores"] for example in examples]),  # shape=(batch_size, n_candidates)
        }

        # Truncate
        if self.max_length is not None:
            output["input_ids"] = output["input_ids"][:, : self.max_length]
            output["attention_mask"] = output["attention_mask"][:, : self.max_length]
            output["labels"] = output["labels"][:, : self.max_length]
        if self.use_logits_to_keep:
            min_prompt_length = min(len(example["prompt_input_ids"]) for example in examples) + 1  # +1 for model ID
            logits_to_keep = max(output["labels"].shape[1] - min_prompt_length, 1)  # Keep at least 1 token
            output["labels"] = output["labels"][:, -logits_to_keep:]
            output["logits_to_keep"] = logits_to_keep
        return output


class CausalLMRouterTrainer(BaseRouterTrainer):
    CollatorType = CausalCollator

    def __init__(
        self,
        model: PreTrainedModel | str,
        processing_class: PreTrainedTokenizerBase,
        model_ids: list[str],
        args: CausalArguments,
        data_collator=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs,
    ):
        self.args = args
        self.model_token_ids: list[int] = processing_class.convert_tokens_to_ids(model_ids)  # type: ignore
        preprocess_args = {
            "fn_kwargs": {
                "tokenizer": processing_class,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
            },
            "batched": False,
            "num_proc": args.dataset_num_proc,
        }
        args.model_init_kwargs["use_fixed_liger"] = args.use_liger_kernel
        if data_collator is None:
            data_collator = CausalCollator(
                processing_class.pad_token_id, self.model_token_ids, args.max_length, args.use_logits_to_keep
            )
        super().__init__(
            model,
            processing_class,
            model_ids,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            preprocess_args,
            **kwargs,
        )

    def plot(self):
        """Plots the training state."""
        assert self.args.output_dir is not None, "output_dir must be set to plot the training state."
        plot_trainer_state(asdict(self.state), ["loss"], self.args.output_dir, "loss")
        plot_trainer_state(asdict(self.state), ["sft_loss"], self.args.output_dir, "sft_loss")
        plot_trainer_state(asdict(self.state), ["accuracy"], self.args.output_dir, "accuracy")
        plot_trainer_state(asdict(self.state), ["reward"], self.args.output_dir, "reward")

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scores = inputs.pop("scores")  # shape=(batch_size, n_candidates)
        if self.args.sft_loss_weight == 0:
            inputs.pop("labels")
        lm_logits, sft_loss, routing_logits = model(**inputs, num_items_in_batch=num_items_in_batch)
        routing_logits = routing_logits.view(-1, self.n_candidates)
        routing_loss = compute_routing_loss(routing_logits, scores, self.args)
        is_training = routing_loss.requires_grad
        prefix = "" if is_training else "eval_"
        metrics = {f"{prefix}routing_loss": routing_loss.detach().cpu()}
        if is_training and self.model_accepts_loss_kwargs:
            routing_loss /= self.args.gradient_accumulation_steps
        if self.args.sft_loss_weight != 0:
            metrics[f"{prefix}sft_loss"] = (
                sft_loss.detach().cpu() * self.args.gradient_accumulation_steps
                if is_training and self.model_accepts_loss_kwargs
                else sft_loss.detach().cpu()
            )
            loss = routing_loss + self.args.sft_loss_weight * sft_loss
        else:
            loss = routing_loss
        if is_training:
            train_reward = scores[range(scores.shape[0]), routing_logits.argmax(dim=1)].mean()
            metrics["reward"] = train_reward.detach().cpu()
        self.store_metrics(metrics, train_eval="train" if is_training else "eval")
        return (loss, routing_logits, lm_logits) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)
        labels = nested_detach(inputs["scores"])
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, routing_logits, lm_logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        routing_logits = nested_detach(routing_logits.view(-1, self.n_candidates))
        return (loss, routing_logits, labels)  # type: ignore

    @override
    def _init_model(self, model: str, model_ids: list[str], pad_token_id: int, **kwargs) -> PreTrainedModel:
        model_instance = load_model_with_reward_head(
            model,
            num_labels=1,
            pad_token_id=pad_token_id,
            use_cache=False,
            **kwargs,
        )
        return model_instance

    @override
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt_input_ids", "response_input_ids", "scores"]

    @staticmethod
    def tokenize_row(
        features: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
    ):
        prompt_input_ids: list[int] = tokenizer.encode(features["prompt"], add_special_tokens=False)  # type: ignore
        response_input_ids: list[list[int]] = [
            tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
            for response in features["responses"]
        ]  # type: ignore

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            response_input_ids = [input_ids[:max_completion_length] for input_ids in response_input_ids]

        return {
            "prompt_input_ids": prompt_input_ids,
            "response_input_ids": response_input_ids,
        }
