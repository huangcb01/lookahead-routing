# Lookahead Routing for Large Language Models

This repository contains the official implementation of the NeurIPS 2025 paper **"[Lookahead Routing for Large Language Models](https://arxiv.org/abs/2510.19506)"**.

## Overview

**Lookahead** is a routing framework that "foresees" potential model outputs by predicting their latent representations and uses these predictions to guide model selection, thus enabling more informed routing without full inference. Within this framework, we implement two approaches based on causal and masked language models. Empirical evaluations across seven public benchmarks — spanning instruction following, mathematical reasoning, and code generation — show that Lookahead consistently outperforms existing routing baselines, achieving an average performance gain of 7.7% over the state-of-the-art.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- PyTorch
- Transformers
- Additional dependencies (see below)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/huangcb01/lookahead-routing.git
cd lookahead-routing
```

2. Install required Python packages:
```bash
pip install torch transformers datasets accelerate deepspeed fire
pip install flash-attn  # Optional, for flash attention support
pip install vllm  # Optional, for vLLM inference engine
pip install liger-kernel  # Optional, for memory-efficient training
```

## Data Preparation

The repository includes compressed training and test datasets in the `compressed_data` directory.

### Decompress Data

To decompress the data files:

```bash
bash decompress_data.sh
```

This will extract the data to the `data` directory with the following structure:
- `data/train/`: Training datasets (UltraFeedback, OpenMathInstruct2, SelfOSSInstruct)
- `data/test/`: Test datasets (AlpacaEval, ArenaHard, MT-Bench, GSM8k, MATH, HumanEval, MBPP)

### Generate Responses (Optional)

If you need to generate new responses from candidate LLMs:

```bash
bash generate_responses.sh
```

Edit the script to specify:
- `MODEL_PATH`: Path to the LLM model
- `DATA_PATH`: Path to input instructions
- `OUTPUT_PATH`: Where to save generated responses

### Compute Rewards (Optional)

To compute reward scores for responses using a reward model:

```bash
bash compute_reward.sh
```

Edit the script to specify:
- `MODEL_NAME_OR_PATH`: Reward model to use (e.g., Skywork-Reward-Gemma-2-27B-v0.2)
- `DATA_DIR`: Directory containing responses
- `LLMS`: List of LLMs to evaluate

## Training

The repository supports multiple router architectures and training methods:

### 1. Lookahead Router with Causal Language Model (CLM)

Train a router based on causal language modeling:

```bash
bash lookahead_clm.sh
```

Key configurations:
- `MODEL_PATH`: Backbone model (e.g., HuggingFaceTB/SmolLM2-135M)
- `TRAIN_SETS`: Training datasets to use
- `TEST_SETS`: Test datasets for evaluation
- `CANDIDATE_LLMS`: List of LLMs to route between
- `LOSS_TYPE`: Loss function (BCE or KL)
- `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`: Training hyperparameters

### 2. Lookahead Router with Masked Language Model (MLM)

Train a router based on masked language modeling:

```bash
bash lookahead_mlm.sh
```

Key configurations:
- `MODEL_PATH`: Backbone model (e.g., answerdotai/ModernBERT-base)
- `MASK_TYPE`: Masking strategy (random or right)
- `MASK_LENGTH`: Length of masked region
- `MASK_WARMUP_RATIO`: Warmup ratio for masking

### 3. Multi-Label Classification (MLC) Router

Train a simple classification-based router:

```bash
bash mlc_router.sh
```

This approach treats routing as a multi-label classification problem.

### 4. Router with Ground Truth Responses

Train a router using ground truth responses:

```bash
bash router_with_gt_responses.sh
```

This is useful for oracle experiments and upper-bound analysis.

## Evaluation

### Baseline Methods

Evaluate baseline routing strategies (oracle, fixed, random):

```bash
bash baselines.sh
```

This will evaluate:
- **Oracle**: Always selects the best LLM (upper bound)
- **Fixed**: Always uses a specific LLM
- **Random**: Randomly selects an LLM

### Custom Evaluation

Use the trained router for evaluation:

```python
python src/eval_baseline.py \
    --router_type <router_type> \
    --candidate_llms "['model1', 'model2', ...]" \
    --data_dir data \
    --test_datasets "['dataset1', 'dataset2', ...]" \
    --output_dir output/eval
```

## Configuration

### Model Configuration

Edit the shell scripts to configure:

- **Backbone Models**: Choose from SmolLM, ModernBERT, or other models
- **Candidate LLMs**: Specify which LLMs to route between
- **Training Parameters**: Batch size, learning rate, epochs, etc.

### DeepSpeed Configuration

For multi-GPU training, DeepSpeed configurations are available in `config/deepspeed/`:
- `ds_z0_config.json`: ZeRO Stage 0
- `ds_z2_config.json`: ZeRO Stage 2
- `ds_z3_config.json`: ZeRO Stage 3

Uncomment the DeepSpeed line in training scripts to enable:
```bash
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --deepspeed config/deepspeed/ds_z3_config.json"
```

### Accelerate Configuration

FSDP and Accelerate configurations are available in `config/`:
- `fsdp.yaml`: Fully Sharded Data Parallel configuration
- `accelerate.yaml`: Accelerate configuration

## Project Structure

```
.
├── src/
│   ├── train.py              # Main training script
│   ├── generate_responses.py # Generate LLM responses
│   ├── compute_rewards.py    # Compute reward scores
│   ├── eval_baseline.py      # Evaluate baseline methods
│   ├── args.py               # Argument parsing
│   ├── metrics.py            # Evaluation metrics
│   ├── model/                # Model implementations
│   ├── trainers/             # Training logic
│   ├── routers/              # Router implementations
│   ├── data/                 # Data loading and processing
│   └── utils/                # Utility functions
├── config/                   # Configuration files
├── compressed_data/          # Compressed datasets
├── lookahead_clm.sh         # Training script for CLM router
├── lookahead_mlm.sh         # Training script for MLM router
├── mlc_router.sh            # Training script for MLC router
├── generate_responses.sh    # Response generation script
├── compute_reward.sh        # Reward computation script
├── baselines.sh             # Baseline evaluation script
└── README.md                # This file
```

## Training Arguments

Common training arguments in `src/train.py`:

- `--backbone_name_or_path`: Pretrained model to use as backbone
- `--data_dir`: Directory containing training data
- `--train_datasets`: List of training datasets
- `--test_datasets`: List of test datasets
- `--candidate_llms`: List of candidate LLMs for routing
- `--loss_type`: Loss function (BCE, KL)
- `--max_length`: Maximum sequence length
- `--per_device_train_batch_size`: Batch size per device
- `--learning_rate`: Learning rate
- `--num_train_epochs`: Number of training epochs
- `--output_dir`: Directory to save model checkpoints

See `src/args.py` for the complete list of arguments.

## Advanced Options

### Flash Attention

Enable Flash Attention 2 for faster training:
```bash
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --attn_implementation flash_attention_2"
```

### Gradient Checkpointing

Enable gradient checkpointing to reduce memory usage:
```bash
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --gradient_checkpointing"
```

### Liger Kernel

Use Liger kernel for memory-efficient training:
```bash
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --use_liger"
```

## Output

Training outputs are saved to the `output` directory with the following structure:
```
output/<train_sets>/<config>/<model>/<method>/<hyperparams>/
├── train.log                 # Training logs
├── train.sh                  # Copy of training script
├── checkpoint-*/             # Model checkpoints
└── eval_results.json         # Evaluation results
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{huang2025lookahead,
  title={Lookahead Routing for Large Language Models},
  author={Canbin Huang and Tianyuan Shi and Yuhua Zhu and Ruijun Chen and Xiaojun Quan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
