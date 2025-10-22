import functools
import os
import transformers
from transformers import set_seed, AutoTokenizer, Trainer

from trainers import *
from args import parse_args
from data.dataset import load_test_datasets, load_train_datasets
from data.utils import get_model_id
from metrics import compute_metrics
from model.utils import resize_embedding_layer
from utils import json2str, get_logger
from utils.file import save_json

logger = get_logger(__name__)
transformers.logging.set_verbosity_info()


def train():
    stage, model_args, data_args, training_args = parse_args()
    set_seed(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Load datasets
    model_ids = [get_model_id(i, data_args.model_id_type) for i in range(len(data_args.candidate_llms))]
    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        with training_args.main_process_first(desc="load dataset"):
            train_dataset, eval_dataset = load_train_datasets(data_args)
        if training_args.do_eval:
            assert eval_dataset is not None, "The evaluation dataset is not provided."
            logger.warning_rank0(
                f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples"
            )
        else:
            logger.warning_rank0(f"Loaded {len(train_dataset)} training examples.")
        logger.warning_rank0(json2str(train_dataset[0]))

    # Initialize trainer
    training_args.model_init_kwargs = {  # type: ignore
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": "bfloat16" if training_args.bf16 else "float32",
    }
    # if training_args.gradient_checkpointing:  # `use_cache=True` is incompatible with gradient checkpointing
    #     training_args.model_init_kwargs["use_cache"] = False
    # if training_args.deepspeed is None:
    #     training_args.model_init_kwargs["device_map"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path)
    if data_args.model_id_type == "special":  # Add model ID tokens
        original_n_tokens = len(tokenizer)
        add_tokens = [get_model_id(i, "special") for i in range(len(data_args.candidate_llms))]
        tokenizer.add_tokens(add_tokens, special_tokens=True)
        new_n_tokens = len(tokenizer)
        logger.warning_rank0(f"{new_n_tokens - original_n_tokens} tokens added to the tokenizer.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    Trainers = {
        "rm": RMRouterTrainer,
        "sc": SCRouterTrainer,
        "causal": CausalLMRouterTrainer,
        "mask": MaskRouterTrainer,
        "mask_concat": MaskConcatRouterTrainer,
    }
    trainer: BaseRouterTrainer = Trainers[stage](
        model=model_args.backbone_name_or_path,
        processing_class=tokenizer,
        model_ids=model_ids,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=(
            functools.partial(compute_metrics, test_set=eval_dataset, tokenizer=tokenizer)
            if eval_dataset is not None
            else None
        ),
    )

    # Train
    if training_args.do_train:
        trainer.train(
            resume_from_checkpoint=(
                True
                if training_args.resume_from_checkpoint == "last-checkpoint"
                else training_args.resume_from_checkpoint
            )
        )
        trainer.save_state()
        trainer.plot()

    # Test
    if training_args.do_predict:
        logger.warning_rank0("Testing the router...")
        all_metrics = {}
        for subset_name, test_dataset in load_test_datasets(data_args):
            logger.warning_rank0(f"Loaded {len(test_dataset)} test examples from {subset_name} subset")
            trainer.compute_metrics = (
                functools.partial(compute_metrics, test_set=test_dataset, dataset_name=subset_name, tokenizer=tokenizer)
                if trainer.is_world_process_zero()
                else None
            )
            _, _, metrics = trainer.predict(test_dataset)
            if trainer.is_world_process_zero():
                predictions = metrics.pop("test_details")
                logger.warning_rank0(f"Metrics for {subset_name}: {json2str(metrics)}")
                save_json(
                    predictions, os.path.join(training_args.output_dir, "test_predictions", f"{subset_name}.json")
                )
            all_metrics[subset_name] = metrics
        trainer.save_metrics("test", all_metrics)


if __name__ == "__main__":
    train()
