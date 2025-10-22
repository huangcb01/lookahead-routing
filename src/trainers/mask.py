import random
import torch
import torch.nn.functional as F
from typing import Any, Literal
from dataclasses import asdict, dataclass
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_pt_utils import nested_detach
from datasets import Dataset
from typing_extensions import override
from trl.trainer.utils import pad

from args import MaskArguments
from utils import plot_trainer_state, get_logger
from model.mask import ModernBertForMaskedLMWithRewardHead
from .base import BaseRouterTrainer
from .utils import compute_routing_loss

logger = get_logger(__name__)


@dataclass
class MaskCollator(DataCollatorMixin):
    cls_token_id: int
    sep_token_id: int
    pad_token_id: int
    mask_token_id: int
    model_token_ids: list[int]
    mask_type: Literal["random", "right", "no"]
    mask_length: int
    starting_mask_rate: float = 0
    packing: bool = False
    warmup_batches: int = 0
    current_batch: int = 0
    return_tensors = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of examples into a batch of tensors.

        Args:
            examples: List of examples.

        Returns:
            A dictionary of 4 items: input_ids, attention_mask, model_token_indices and rewards.
        """
        # Concatenate prompt, model ID and response
        n_candidates = len(self.model_token_ids)
        mask_rate = (
            min(self.starting_mask_rate + (1 - self.starting_mask_rate) * self.current_batch / self.warmup_batches, 1)
            if self.warmup_batches != 0
            else 1.0
        )
        masked_responses = [
            self.mask_response(response_ids, mask_rate)
            for example in examples
            for response_ids in example["response_input_ids"]
        ]
        input_ids = [
            torch.tensor(
                [
                    self.cls_token_id,
                    *example["prompt_input_ids"],
                    self.sep_token_id,
                    self.model_token_ids[j],
                    *(masked_responses[i * n_candidates + j][0]),
                    self.sep_token_id,
                ]
            )
            for i, example in enumerate(examples)
            for j in range(n_candidates)
        ]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        seqlens = torch.tensor([len(ids) for ids in input_ids])

        if self.packing:  # Concatenate all examples into a one-dimensional tensor
            output: dict[str, Tensor | int | float] = {
                "input_ids": torch.cat(input_ids, dim=0),  # shape=(batch_size*n_candidates*seq_length,)
                "attention_mask": torch.cat(attention_mask, dim=0),  # shape=(batch_size*n_candidates*seq_length,)
                "cu_seqlens": F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)),
                "max_seqlen": int(seqlens.max().item()),
                "batch_size": len(examples) * n_candidates,
                "seq_len": int(seqlens.max().item()),
            }
        else:  # Pad each example to the maximum length
            output = {
                "input_ids": pad(
                    input_ids, padding_value=self.pad_token_id, padding_side="left"
                ),  # shape=(batch_size*n_candidates, seq_length)
                "attention_mask": pad(
                    attention_mask, padding_value=0, padding_side="left"
                ),  # shape=(batch_size*n_candidates, seq_length)
            }
        if self.mask_type != "no":
            labels = [torch.tensor(masked_response[1]) for masked_response in masked_responses]
            output["labels"] = pad(labels, padding_value=-100)  # shape=(batch_size*n_candidates, seq_length)
        output["scores"] = torch.tensor(
            [example["scores"] for example in examples], dtype=torch.float32
        )  # shape=(batch_size, n_candidates)
        output["mask_rate"] = mask_rate
        self.current_batch += 1
        return output

    def mask_response(self, response_ids: list[int], mask_rate: float):
        if self.mask_length == 0:
            return [], []
        if self.mask_type == "random":
            masked_response_ids = [
                self.mask_token_id if random.random() < mask_rate else token for token in response_ids
            ]
            masked_response_ids += [self.mask_token_id] * (self.mask_length - len(masked_response_ids))
        elif self.mask_type == "right":
            mask_length = max(int(len(response_ids) * mask_rate), 1)
            masked_response_ids = response_ids[:-mask_length] + [self.mask_token_id] * (
                self.mask_length - len(response_ids) + mask_length
            )
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
        labels = [
            -100 if original_id == masked_id else original_id
            for original_id, masked_id in zip(response_ids, masked_response_ids)
        ]
        return masked_response_ids, labels


class MaskRouterTrainer(BaseRouterTrainer):
    CollatorType = MaskCollator

    def __init__(
        self,
        model: PreTrainedModel | str,
        processing_class: PreTrainedTokenizerBase,
        model_ids: list[str],
        args: MaskArguments,
        data_collator=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs,
    ):
        self.args = args
        self.model_token_ids: list[int] = processing_class.convert_tokens_to_ids(model_ids)  # type: ignore
        self._data_collator_args = {
            "cls_token_id": processing_class.cls_token_id,
            "sep_token_id": processing_class.sep_token_id,
            "pad_token_id": processing_class.pad_token_id,
            "mask_token_id": processing_class.mask_token_id,
            "model_token_ids": self.model_token_ids,
            "mask_type": args.mask_type,
            "mask_length": args.mask_length,
            "starting_mask_rate": args.starting_mask_rate,
            "packing": args.model_init_kwargs["attn_implementation"] == "flash_attention_2",
        }
        preprocess_args = {
            "fn_kwargs": {
                "tokenizer": processing_class,
                "response_truncation_side": args.response_truncation_side,
                "max_prompt_length": args.max_length - args.mask_length - 3,
                "max_completion_length": args.mask_length,
            },
            "batched": False,
            "num_proc": args.dataset_num_proc,
        }
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

    @override
    def plot(self):
        """Plots the training state."""
        assert self.args.output_dir is not None, "output_dir must be set to plot the training state."
        plot_trainer_state(asdict(self.state), ["loss"], self.args.output_dir, "loss")
        plot_trainer_state(asdict(self.state), ["routing_loss"], self.args.output_dir, "routing_loss")
        plot_trainer_state(asdict(self.state), ["mlm_loss"], self.args.output_dir, "mlm_loss")
        # plot_trainer_state(asdict(self.state), ["contrast_loss"], self.args.output_dir, "contrast_loss")
        plot_trainer_state(asdict(self.state), ["reward"], self.args.output_dir, "reward")
        plot_trainer_state(asdict(self.state), ["mask_rate"], self.args.output_dir, "mask_rate")
        plot_trainer_state(asdict(self.state), ["accuracy"], self.args.output_dir, "accuracy")

    @override
    def get_train_dataloader(self) -> DataLoader:
        """Override to use different collator for training and evaluation."""
        if self.args.num_train_epochs is not None:
            assert isinstance(self.train_dataset, Dataset)
            total_batches = len(self.train_dataset) / self.args.per_device_train_batch_size * self.args.num_train_epochs
        else:
            total_batches = self.args.max_steps * self.args.gradient_accumulation_steps
        warmup_batches = int(total_batches * self.args.mask_warmup_ratio)
        logger.warning_rank0(f"Warmup batches: {warmup_batches}")
        current_batch = self.state.global_step * self.args.gradient_accumulation_steps
        self.data_collator = self.CollatorType(
            **self._data_collator_args, warmup_batches=warmup_batches, current_batch=current_batch
        )
        data_loader = super().get_train_dataloader()
        self.data_collator = None
        return data_loader

    @override
    def get_eval_dataloader(self, eval_dataset: str | TorchDataset | None = None) -> DataLoader:
        """Override to use different collator for training and evaluation."""
        self.data_collator = self.CollatorType(**self._data_collator_args, warmup_batches=0)
        data_loader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = None
        return data_loader

    @override
    def get_test_dataloader(self, test_dataset: TorchDataset) -> DataLoader:
        self.data_collator = self.CollatorType(**self._data_collator_args, warmup_batches=0)
        data_loader = super().get_test_dataloader(test_dataset)
        self.data_collator = None
        return data_loader

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scores = inputs.pop("scores")
        mask_rate = float(inputs.pop("mask_rate"))
        if self.args.mlm_loss_weight == 0:
            inputs.pop("labels", None)  # pop labels to avoid loss calculation
        if self.model_accepts_loss_kwargs:
            inputs["num_items_in_batch"] = num_items_in_batch
        mlm_loss, routing_logits, output_token_ids = model(**inputs, mask_length=self.args.mask_length)
        routing_logits = routing_logits.view(-1, self.n_candidates)
        loss = compute_routing_loss(routing_logits, scores, self.args)
        is_training = loss.requires_grad
        prefix = "" if is_training else "eval_"
        metrics = {f"{prefix}mask_rate": mask_rate, f"{prefix}routing_loss": loss.detach().cpu()}
        if self.args.mlm_loss_weight != 0:
            metrics[f"{prefix}mlm_loss"] = mlm_loss.detach().cpu()
            loss += self.args.mlm_loss_weight * mlm_loss
        # if self.args.contrast_loss_weight != 0:
        #     contrast_loss = self._compute_contrast_loss(hidden_states, labels, scores)
        #     metrics[f"{prefix}contrast_loss"] = (
        #         contrast_loss.detach().cpu() if isinstance(contrast_loss, Tensor) else contrast_loss
        #     )
        #     loss += self.args.contrast_loss_weight * contrast_loss
        if is_training:
            train_reward = scores[range(scores.shape[0]), routing_logits.argmax(dim=1)].mean()
            metrics["reward"] = train_reward.detach().cpu()
        self.store_metrics(metrics, train_eval="train" if is_training else "eval")
        return (loss, routing_logits, output_token_ids) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        inputs = self._prepare_inputs(inputs)
        labels = nested_detach(inputs["scores"])
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, routing_logits, output_token_ids = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        routing_logits = nested_detach(routing_logits.view(-1, self.n_candidates))
        output_token_ids = nested_detach(output_token_ids.view(*routing_logits.shape, -1))
        return (loss, (routing_logits, output_token_ids), labels)  # type: ignore

    @override
    def _init_model(self, model: str, model_ids: list[str], pad_token_id: int, **kwargs) -> PreTrainedModel:
        model_instance = ModernBertForMaskedLMWithRewardHead.from_pretrained(model, pad_token_id=pad_token_id, **kwargs)
        # model_instance.init_reward_head()
        return model_instance

    @override
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt_input_ids", "response_input_ids", "scores"]

    def _compute_contrast_loss(self, hidden_states: Tensor, labels: Tensor, scores: Tensor):
        batch_size = scores.shape[0]
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.view(
            batch_size, self.n_candidates, -1, hidden_size
        )  # shape=(batch_size, n_candidates, seq_length, hidden_size)
        labels = labels.view(batch_size, self.n_candidates, -1)  # shape=(batch_size, n_candidates, seq_length)
        positive_mask = (
            scores >= scores.max(dim=1, keepdim=True).values * self.args.bce_threshold
        )  # shape=(batch_size, n_candidates)

        loss, n_items = 0, 0
        for i in range(batch_size):
            positive_hidden_states = hidden_states[i][positive_mask[i]].transpose(
                0, 1
            )  # shape=(seq_length, num_positive, hidden_size)
            negative_hidden_states = hidden_states[i][~(positive_mask[i])].transpose(
                0, 1
            )  # shape=(seq_length, num_negative, hidden_size)
            distance = (
                torch.cdist(positive_hidden_states, negative_hidden_states, p=1) / hidden_size
            )  # shape=(seq_length, num_positive, num_negative)
            positive_labels = (
                labels[i][positive_mask[i]].transpose(0, 1).unsqueeze(2)
            )  # shape=(seq_length, num_positive, 1)
            negative_labels = (
                labels[i][~(positive_mask[i])].transpose(0, 1).unsqueeze(1)
            )  # shape=(seq_length, 1, num_negative)
            mask = (
                (positive_labels != -100) & (negative_labels != -100) & (positive_labels != negative_labels)
            )  # shape=(seq_length, num_positive, num_negative)
            loss += distance[mask].sum()
            n_items += mask.sum()
        return -loss / n_items if n_items > 0 else 0

    @staticmethod
    def tokenize_row(
        features: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        response_truncation_side: Literal["left", "right"] = "right",
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
    ):
        prompt_input_ids: list[int] = tokenizer.encode(features["prompt"], add_special_tokens=False)  # type: ignore
        response_input_ids: list[list[int]] = [
            tokenizer.encode(response, add_special_tokens=False) for response in features["responses"]
        ]  # type: ignore

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            if response_truncation_side == "right":
                response_input_ids = [input_ids[:max_completion_length] for input_ids in response_input_ids]
            else:
                response_input_ids = [input_ids[-max_completion_length:] for input_ids in response_input_ids]

        return {
            "prompt_input_ids": prompt_input_ids,
            "response_input_ids": response_input_ids,
        }
