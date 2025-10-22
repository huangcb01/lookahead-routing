import torch
import torch.nn as nn
from typing import Any
from dataclasses import dataclass, asdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset
from typing_extensions import override
from trl.trainer.utils import pad

from args import RMArguments
from utils import plot_trainer_state, get_logger
from .utils import compute_routing_loss
from .base import BaseRouterTrainer

logger = get_logger(__name__)


@dataclass
class RMCollator(DataCollatorMixin):
    pad_token_id: int
    max_length: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of examples into a batch of tensors.

        Args:
            examples: List of examples.

        Returns:
            A dictionary of 4 items: input_ids, attention_mask, model_token_indices and rewards.
        """
        # Concatenate prompt and response
        input_ids = [
            torch.tensor(example["prompt_input_ids"] + response_input_ids)
            for example in examples
            for response_input_ids in example["response_input_ids"]
        ]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        # Pad
        output = {
            "input_ids": pad(input_ids, padding_value=self.pad_token_id),  # shape=(batch_size*n_candidates, seq_length)
            "attention_mask": pad(attention_mask, padding_value=0),  # shape=(batch_size*n_candidates, seq_length)
            "scores": torch.tensor([example["scores"] for example in examples]),  # shape=(batch_size, n_candidates)
        }
        if self.max_length is not None:
            output["input_ids"] = output["input_ids"][:, : self.max_length]
            output["attention_mask"] = output["attention_mask"][:, : self.max_length]
        return output


class RMRouterTrainer(BaseRouterTrainer):
    CollatorType = RMCollator

    def __init__(
        self,
        model: PreTrainedModel | str,
        processing_class: PreTrainedTokenizerBase,
        model_ids: list[str],
        args: RMArguments,
        data_collator=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs,
    ):
        self.args = args
        self.model_ids = model_ids
        preprocess_args = {
            "fn_kwargs": {
                "tokenizer": processing_class,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
            },
            "batched": False,
            "num_proc": args.dataset_num_proc,
        }
        if data_collator is None:
            data_collator = RMCollator(processing_class.pad_token_id, args.max_length)
        args.label_names = ["scores"]
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
        plot_trainer_state(asdict(self.state), ["loss", "reward_loss", "sft_loss"], self.args.output_dir, "loss")
        plot_trainer_state(asdict(self.state), ["accuracy"], self.args.output_dir, "accuracy")
        plot_trainer_state(asdict(self.state), ["reward"], self.args.output_dir, "reward")

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scores = inputs.pop("scores")  # shape=(batch_size, n_candidates)
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, self.n_candidates)  # shape=(batch_size*n_candidates, n_candidates)
        loss = compute_routing_loss(logits, scores, self.args)
        is_training = loss.requires_grad
        if is_training and self.model_accepts_loss_kwargs:
            loss /= self.args.gradient_accumulation_steps
        metrics = {}
        if is_training:
            train_reward = scores[range(scores.shape[0]), logits.argmax(dim=1)].mean()
            metrics["reward"] = train_reward.detach().cpu()
        self.store_metrics(metrics, train_eval="train" if is_training else "eval")
        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return loss, logits.view(-1, self.n_candidates) if logits is not None else None, labels

    @override
    def _init_model(self, model: str, model_ids: list[str], pad_token_id: int, **kwargs) -> PreTrainedModel:
        model_instance = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=1,
            pad_token_id=pad_token_id,
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
            tokenizer.encode(response, add_special_tokens=False)
            for response in features["responses"]
        ]  # type: ignore

        if isinstance(tokenizer.cls_token_id, int) and isinstance(tokenizer.sep_token_id, int):
            prompt_input_ids = [tokenizer.cls_token_id] + prompt_input_ids + [tokenizer.sep_token_id]
            response_input_ids = [
                input_ids + [tokenizer.sep_token_id] for input_ids in response_input_ids
            ]
        elif isinstance(tokenizer.eos_token_id, int):
            response_input_ids = [
                input_ids + [tokenizer.eos_token_id] for input_ids in response_input_ids
            ]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            response_input_ids = [input_ids[:max_completion_length] for input_ids in response_input_ids]

        return {
            "prompt_input_ids": prompt_input_ids,
            "response_input_ids": response_input_ids,
        }
