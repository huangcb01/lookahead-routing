import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Literal
from transformers import TrainingArguments, HfArgumentParser

from utils import save_json, get_logger

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    backbone_name_or_path: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct", metadata={"help": "Name or path of the backbone model."}
    )
    attn_implementation: str = field(
        default="sdpa", metadata={"help": "Attention implementation.", "choices": ["sdpa", "flash_attention_2"]}
    )


@dataclass
class DataArguments:
    data_dir: str = field(metadata={"help": "Path to the training data directory."})
    train_datasets: list[str] = field(
        default_factory=list,
        metadata={"help": "List of training datasets, each of which is a path relative to `data_dir/train`."},
    )
    test_datasets: list[str] = field(default_factory=list, metadata={"help": "List of test datasets."})
    train_slice: tuple[float, float] = field(
        default=(0, 1), metadata={"help": "Slice of the training set to use. E.g. (0, 0.5) to use the first half."}
    )
    max_train_samples: int | None = field(default=None, metadata={"help": "Maximum number of training samples to use."})
    candidate_llms: list[str] = field(default_factory=list, metadata={"help": "List of candidate LLMs."})
    model_id_type: str = field(
        default="letter", metadata={"help": "Type of model id.", "choices": ["letter", "number", "special", "unused"]}
    )
    score_normalization: str = field(
        default="none",
        metadata={"help": "Type of score normalization.", "choices": ["none", "min-max", "softmax", "standardization"]},
    )
    cluster_id_name: str | None = field(default=None, metadata={"help": "Name of the cluster id in the dataset."})
    tag_score_weight: float = field(default=0, metadata={"help": "Weight of the tag-level score."})

    def __post_init__(self):
        # Fix tuple parsing
        if len(self.train_slice) != 2:
            s = "".join(self.train_slice)
            self.train_slice = eval(s)
            assert isinstance(self.train_slice, tuple) and len(self.train_slice) == 2, f"Invalid train_slice: {s}"


@dataclass
class SCArguments(TrainingArguments):
    loss_type: Literal["CE", "BCE", "MSE", "ForwardKL", "ReverseKL"] = field(
        default="CE", metadata={"help": "Loss type for the sequence classification router."}
    )
    kl_temperature: float = field(default=1)
    bce_threshold: float = field(default=0.5)
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    dataset_num_proc: int = field(default=1, metadata={"help": "Number of processes to use for dataset loading."})
    model_init_kwargs: dict[str, Any] = field(default_factory=dict, metadata={"help": "Do no specify this."})


@dataclass
class ContrastArguments(SCArguments):
    similarity_function: Literal["cos", "dot"] = field(
        default="cos",
        metadata={
            "help": "Similarity function to use for contrastive learning.",
            "choices": ["cos", "dot"],
        },
    )
    sample_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the sample-sample contrast loss."})
    top_k_llms: int = field(default=3, metadata={"help": "Number of top-k similar LLMs to use."})
    last_k_llms: int = field(default=3, metadata={"help": "Number of last-k similar LLMs to use."})
    last_k_samples: int = field(default=3, metadata={"help": "Number of last-k similar samples to use."})


@dataclass
class MaskArguments(SCArguments):
    mlm_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the SFT loss."})
    # contrast_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the contrast loss."})
    mask_type: Literal["random", "right", "no", "left"] = field(
        default="random", metadata={"help": "Random mask or mask from the right side.", "choices": ["random", "right"]}
    )
    mask_length: int = field(default=1024, metadata={"help": "Maximum length of the response."})
    mask_warmup_ratio: float = field(default=0, metadata={"help": "Warmup ratio for the mask probability."})
    starting_mask_rate: float = field(default=0, metadata={"help": "Starting mask rate."})
    response_truncation_side: Literal["left", "right"] = field(
        default="right", metadata={"help": "Truncation side of the response.", "choices": ["left", "right"]}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.mask_type == "no":
            self.mask_warmup_ratio = 0
            self.starting_mask_rate = 0
            self.mlm_loss_weight = 0
            # self.contrast_loss_weight = 0


@dataclass
class NCAArguments(SCArguments):
    precompute_ref_log_probs: bool = field(
        default=True, metadata={"help": "Whether to precompute reference log probabilities."}
    )
    ref_log_prob_path: str | None = field(default=None, metadata={"help": "Path to cache reference log probabilities."})
    nca_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the NCA loss."})
    nca_temperature_alpha: float = field(
        default=1e-4,
        metadata={
            "help": "The temperature_alpha factor in NCA loss. Higher temperature_alpha means less divergence from the initial policy."
        },
    )
    nca_beta: float = field(
        default=0.1,
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model."
        },
    )
    nca_loss_type: Literal["NCA", "InfoNCA"] = field(
        default="NCA",
        metadata={"help": "NCA loss type"},
    )
    use_logits_to_keep: bool = field(default=True, metadata={"help": "Use `logits_to_keep` to save memory."})
    max_prompt_length: int | None = field(default=None, metadata={"help": "Maximum length of the prompt."})
    max_completion_length: int | None = field(default=None, metadata={"help": "Maximum length of the response."})


@dataclass
class CausalArguments(SCArguments):
    sft_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the SFT loss."})
    use_logits_to_keep: bool = field(default=True, metadata={"help": "Use `logits_to_keep` to save memory."})
    max_prompt_length: int | None = field(default=None, metadata={"help": "Maximum length of the prompt."})
    max_completion_length: int | None = field(default=None, metadata={"help": "Maximum length of the response."})

    def __post_init__(self):
        super().__post_init__()
        if self.sft_loss_weight == 0:
            logger.warning_rank0(
                "Setting `max_completion_length` to 1 (for only EOS token) since `sft_loss_weight` is 0, which disables the SFT loss."
            )
            self.max_completion_length = 1
        if self.use_liger_kernel:
            logger.warning_rank0("Setting `use_logits_to_keep` to False since `use_liger_kernel` is True.")
            self.use_logits_to_keep = False


@dataclass
class RDArguments(TrainingArguments):
    loss_type: Literal["MSE", "CE", "ForwardKL", "ReverseKL"] = field(
        default="MSE", metadata={"help": "Loss type for the reward prediction."}
    )
    temperature: float = field(default=1)
    distill_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the distillation loss."})
    reward_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the reward modeling loss."})
    use_logits_to_keep: bool = field(default=True, metadata={"help": "Use `logits_to_keep` to save memory."})
    max_prompt_length: int | None = field(default=None, metadata={"help": "Maximum length of the prompt."})
    max_completion_length: int | None = field(default=None, metadata={"help": "Maximum length of the response."})
    max_length: int = field(
        default=2048, metadata={"help": "Maximum length of the sequences (prompt + completion) in the batch."}
    )
    dataset_num_proc: int = field(default=1, metadata={"help": "Number of processes to use for dataset loading."})
    model_init_kwargs: dict[str, Any] = field(default_factory=dict, metadata={"help": "Do no specify this."})

    def __post_init__(self):
        super().__post_init__()
        if self.use_liger_kernel:
            logger.warning_rank0("Setting `use_logits_to_keep` to False since `use_liger_kernel` is True.")
            self.use_logits_to_keep = False


@dataclass
class ReconstructArguments(TrainingArguments):
    loss_type: Literal["MSE", "CE", "ForwardKL", "ReverseKL"] = field(
        default="MSE", metadata={"help": "Loss type for the reward prediction."}
    )
    temperature: float = field(default=1)
    reconstruct_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the SFT loss."})
    use_logits_to_keep: bool = field(default=True, metadata={"help": "Use `logits_to_keep` to save memory."})
    max_prompt_length: int | None = field(default=None, metadata={"help": "Maximum length of the prompt."})
    max_completion_length: int | None = field(default=None, metadata={"help": "Maximum length of the response."})
    max_length: int = field(
        default=2048, metadata={"help": "Maximum length of the sequences (prompt + completion) in the batch."}
    )
    dataset_num_proc: int = field(default=1, metadata={"help": "Number of processes to use for dataset loading."})
    model_init_kwargs: dict[str, Any] = field(default_factory=dict, metadata={"help": "Do no specify this."})

    def __post_init__(self):
        super().__post_init__()
        if self.reconstruct_loss_weight == 0:
            logger.warning_rank0(
                "Setting `max_completion_length` to 1 (for only EOS token) since `sft_loss_weight` is 0, which disables the SFT loss."
            )
            self.max_completion_length = 1
        if self.use_liger_kernel:
            logger.warning_rank0("Setting `use_logits_to_keep` to False since `use_liger_kernel` is True.")
            self.use_logits_to_keep = False


@dataclass
class TokenPredictionArguments(TrainingArguments):
    loss_type: Literal["MSE", "CE", "ForwardKL", "ReverseKL"] = field(
        default="MSE", metadata={"help": "Loss type for the reward prediction."}
    )
    temperature: float = field(default=1)
    token_loss_weight: float = field(default=0.5, metadata={"help": "Weight of the token prediction loss."})
    max_length: int = field(default=2048, metadata={"help": "Maximum length of the prompt in the batch."})
    good_threshold: float = field(default=1.0, metadata={"help": "Threshold for good responses."})
    dataset_num_proc: int = field(default=1, metadata={"help": "Number of processes to use for dataset loading."})
    model_init_kwargs: dict[str, Any] = field(default_factory=dict, metadata={"help": "Do no specify this."})

    def __post_init__(self):
        super().__post_init__()
        if self.use_liger_kernel:
            logger.warning_rank0("Setting `use_logits_to_keep` to False since `use_liger_kernel` is True.")
            self.use_logits_to_keep = False


@dataclass
class RMArguments(SCArguments):
    max_prompt_length: int | None = field(default=None, metadata={"help": "Maximum length of the prompt."})
    max_completion_length: int | None = field(default=None, metadata={"help": "Maximum length of the response."})


ARGUMENT_CLASSES = {
    "rm": RMArguments,
    "sc": SCArguments,
    "contrast": ContrastArguments,
    "mask": MaskArguments,
    "mask_concat": MaskArguments,
    "causal": CausalArguments,
    "rd": RDArguments,
    "reconstruct": ReconstructArguments,
    "token": TokenPredictionArguments,
    "bert_nca": NCAArguments,
    "causal_nca": NCAArguments,
}


def parse_args() -> tuple[str, ModelArguments, DataArguments, TrainingArguments]:
    """Parses the arguments.

    Returns:
        The stage, model arguments, data arguments and training arguments.
    """
    stage = sys.argv[1]
    del sys.argv[1]
    assert stage in ARGUMENT_CLASSES.keys(), f"The first argument must be one of {set(ARGUMENT_CLASSES.keys())}."
    parser = HfArgumentParser([ModelArguments, DataArguments, ARGUMENT_CLASSES[stage]])  # type: ignore
    router_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    all_args = {
        "router": asdict(router_args),
        "data": asdict(data_args),
        "training": training_args.to_dict(),
        "remaining_args": remaining_args,
    }
    save_json(all_args, os.path.join(training_args.output_dir, "args.json"))
    assert not remaining_args, f"Unrecognized arguments: {remaining_args}"
    return stage, router_args, data_args, training_args
