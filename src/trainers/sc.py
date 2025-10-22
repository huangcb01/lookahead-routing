from typing import Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from datasets import Dataset
from typing_extensions import override

from args import SCArguments
from utils import get_logger
from .base import BaseRouterTrainer
from .utils import compute_routing_loss

logger = get_logger(__name__)


class SCRouterTrainer(BaseRouterTrainer):
    def __init__(
        self,
        model: PreTrainedModel | str,
        processing_class: PreTrainedTokenizerBase,
        model_ids: list[str],
        args: SCArguments,
        data_collator=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs,
    ):
        preprocess_args = {
            "fn_kwargs": {
                "tokenizer": processing_class,
                "max_length": args.max_length,
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

    def _init_model(self, model: str, model_ids: list[str], pad_token_id: int, **kwargs) -> PreTrainedModel:
        id2label = {i: label for i, label in enumerate(model_ids)}
        label2id = {label: i for i, label in enumerate(model_ids)}
        model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=len(model_ids), id2label=id2label, label2id=label2id, pad_token_id=pad_token_id, **kwargs
        )
        assert isinstance(model, PreTrainedModel)
        return model

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scores = inputs.pop("scores")
        if self.model_accepts_loss_kwargs:
            inputs["num_items_in_batch"] = num_items_in_batch
        outputs = model(**inputs)
        loss = compute_routing_loss(outputs["logits"], scores, self.args)
        if loss.requires_grad and self.model_accepts_loss_kwargs:
            loss /= self.args.gradient_accumulation_steps
        outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def tokenize_row(features: dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_length: int | None = None):
        return tokenizer(features["prompt"], truncation=True, max_length=max_length)
