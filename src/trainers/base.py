import os
import torch
from collections import UserDict, defaultdict
from typing import Any, Literal
from dataclasses import asdict
from torch.utils.data import DataLoader
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from datasets import Dataset
from typing_extensions import override

from args import SCArguments
from model.utils import resize_embedding_layer
from utils import plot_trainer_state, get_logger
from utils.file import save_json
from utils.others import json2str

logger = get_logger(__name__)


class BaseRouterTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | str,
        processing_class: PreTrainedTokenizerBase,
        model_ids: list[str],
        args: SCArguments,
        data_collator=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        preprocess_args: dict[str, Any] = {},
        **kwargs,
    ):
        self.args = args
        self.model_ids = model_ids
        self.n_candidates = len(model_ids)
        self.preprocess_args: dict[str, Any] = preprocess_args
        if isinstance(model, str):
            model = self._init_model(model, model_ids, processing_class.pad_token_id, **args.model_init_kwargs)  # type: ignore
        assert isinstance(model, PreTrainedModel)
        resize_embedding_layer(model, processing_class)
        with args.main_process_first(desc="preprocess dataset"):
            if train_dataset is not None:
                train_dataset = train_dataset.map(self.tokenize_row, **self.preprocess_args)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, **self.preprocess_args)
        args.label_names = ["scores"]
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, **kwargs)

        # print data
        item = dict(next(iter(self.get_train_dataloader())))
        for k in list(item.keys()):
            if isinstance(item[k], torch.Tensor):
                item[k] = item[k].tolist()
            if isinstance(item[k], UserDict):
                item[k] = dict(item[k])
            if "ids" in k:
                item[k.replace("ids", "tokens")] = [processing_class.convert_ids_to_tokens(x) for x in item[k]]
        logger.warning_rank0(json2str(item))

    def _init_model(self, model: str, model_ids: list[str], pad_token_id: int, **kwargs) -> PreTrainedModel:
        raise NotImplementedError("Subclasses must implement _init_model method.")

    def plot(self):
        """Plots the training state."""
        assert self.args.output_dir is not None, "output_dir must be set to plot the training state."
        plot_trainer_state(asdict(self.state), ["loss"], self.args.output_dir, "loss")
        plot_trainer_state(asdict(self.state), ["reward"], self.args.output_dir, "reward")

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        super().log(logs, start_time)

    @override
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Override to save predictions in the evaluation."""
        eval_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        if metric_key_prefix == "eval" and self.args.output_dir is not None:
            details: list[dict] = eval_output.metrics.pop("eval_details")  # type: ignore
            save_json(details, os.path.join(self.args.output_dir, "eval_details", f"{self.state.global_step}.json"))
        return eval_output

    @override
    def predict(
        self, test_dataset: Dataset, ignore_keys: list[str] | None = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        with self.args.main_process_first(desc="preprocess dataset"):
            test_dataset = test_dataset.map(self.tokenize_row, **self.preprocess_args)
        torch.cuda.empty_cache()
        outputs = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        return outputs

    @staticmethod
    def tokenize_row(features: dict[str, Any], tokenizer: PreTrainedTokenizerBase, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement tokenize_row method.")
