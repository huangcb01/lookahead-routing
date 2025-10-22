import random
import torch
import torch.nn.functional as F
from typing import Any
from torch import Tensor
from transformers import PreTrainedModel
from typing_extensions import override
from trl.trainer.utils import pad

from utils import get_logger
from model.mask_concat import ModernBertForMaskedConcatLMWithRewardHead
from .mask import MaskCollator, MaskRouterTrainer

logger = get_logger(__name__)


class MaskConcatCollator(MaskCollator):
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
        masked_responses = [self.mask_responses(example["response_input_ids"], mask_rate) for example in examples]
        input_ids = [
            torch.tensor([self.cls_token_id, *example["prompt_input_ids"], self.sep_token_id, *(masked_response[0])])
            for example, masked_response in zip(examples, masked_responses)
        ]
        position_ids = [
            torch.tensor(
                list(range(prompt_len := len(example["prompt_input_ids"]) + 2))
                + list(range(prompt_len, prompt_len + self.mask_length + 1)) * n_candidates
            )
            for example in examples
        ]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        seqlens = torch.tensor([len(ids) for ids in input_ids])

        if self.packing:  # Concatenate all examples into a one-dimensional tensor
            output: dict[str, Tensor | int | float] = {
                "input_ids": torch.cat(input_ids, dim=0),  # shape=(batch_size*n_candidates*seq_length,)
                "attention_mask": torch.cat(attention_mask, dim=0),  # shape=(batch_size*n_candidates*seq_length,)
                "cu_seqlens": F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)),
                "max_seqlen": int(seqlens.max().item()),
                "seq_len": int(seqlens.max().item()),
            }
        else:  # Pad each example to the maximum length
            output = {
                "input_ids": pad(
                    input_ids, padding_value=self.pad_token_id, padding_side="left"
                ),  # shape=(batch_size*n_candidates, seq_length)
                "position_ids": pad(
                    position_ids, padding_value=0, padding_side="left"
                ),  # shape=(batch_size*n_candidates, seq_length)
                "attention_mask": pad(
                    attention_mask, padding_value=0, padding_side="left"
                ),  # shape=(batch_size*n_candidates, seq_length)
                "seq_lens": seqlens,  # shape=(batch_size,)
            }
        if self.mask_type != "no":
            labels = [torch.tensor(masked_response[1]) for masked_response in masked_responses]
            output["labels"] = pad(labels, padding_value=-100)  # shape=(batch_size*n_candidates, seq_length)
        output["scores"] = torch.tensor(
            [example["scores"] for example in examples], dtype=torch.float32
        )  # shape=(batch_size, n_candidates)
        output["mask_rate"] = mask_rate
        output["batch_size"] = len(examples)
        self.current_batch += 1
        return output

    def mask_responses(self, responses_ids: list[list[int]], mask_rate: float):
        if self.mask_length == 0:
            return [], []
        if self.mask_type == "random":
            masked_responses_ids = [
                [model_token_id if random.random() < mask_rate else token for token in response_ids]
                + [model_token_id] * (self.mask_length - len(response_ids))
                + [self.sep_token_id]
                for model_token_id, response_ids in zip(self.model_token_ids, responses_ids)
            ]
            labels = [
                [
                    -100 if original_id == masked_id else original_id
                    for original_id, masked_id in zip(response_ids, masked_response_ids)
                ]
                + [-100] * (self.mask_length - len(response_ids) + 1)
                for response_ids, masked_response_ids in zip(responses_ids, masked_responses_ids)
            ]
            # flatten
            masked_responses_ids = [i for masked_response_ids in masked_responses_ids for i in masked_response_ids]
            labels = [i for label in labels for i in label]
        elif self.mask_type == "right":
            masked_responses_ids, labels = [], []
            for model_token_id, response_ids in zip(self.model_token_ids, responses_ids):
                mask_length = max(int(len(response_ids) * mask_rate), 1)
                masked_response_ids = (
                    response_ids[:-mask_length]
                    + [model_token_id] * (self.mask_length - len(response_ids) + mask_length)
                    + [self.sep_token_id]
                )
                label = (
                    [-100] * (len(response_ids) - mask_length)
                    + response_ids[-mask_length:]
                    + [-100] * (self.mask_length - len(response_ids) + 1)
                )
                masked_responses_ids.extend(masked_response_ids)
                labels.extend(label)
        elif self.mask_type == "left":
            masked_responses_ids, labels = [], []
            for model_token_id, response_ids in zip(self.model_token_ids, responses_ids):
                mask_length = max(int(len(response_ids) * mask_rate), 1)
                num_padding = self.mask_length - len(response_ids)
                masked = (
                    [model_token_id] * num_padding
                    + [model_token_id] * mask_length
                    + response_ids[mask_length:]
                    + [self.sep_token_id]
                )
                label = (
                    [-100] * num_padding
                    + response_ids[:mask_length]
                    + [-100] * (len(response_ids) - mask_length)
                    + [-100]
                )
                masked_responses_ids.extend(masked)
                labels.extend(label)
        elif self.mask_type == "no":
            masked_responses_ids, labels = [], []
            for response_ids in responses_ids:
                masked_responses_ids.extend(response_ids + [self.sep_token_id])
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
        return masked_responses_ids, labels


class MaskConcatRouterTrainer(MaskRouterTrainer):
    CollatorType = MaskConcatCollator

    @override
    def _init_model(self, model: str, model_ids: list[str], pad_token_id: int, **kwargs) -> PreTrainedModel:
        id2label = {i: label for i, label in enumerate(model_ids)}
        label2id = {label: i for i, label in enumerate(model_ids)}
        model_instance = ModernBertForMaskedConcatLMWithRewardHead.from_pretrained(
            model, num_labels=len(model_ids), id2label=id2label, label2id=label2id, pad_token_id=pad_token_id, **kwargs
        )
        # model_instance.init_reward_head()
        return model_instance
