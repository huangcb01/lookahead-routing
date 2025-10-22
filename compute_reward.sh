set -e
set -o pipefail

MODEL_NAME_OR_PATH=Skywork/Skywork-Reward-Gemma-2-27B-v0.2
DATA_DIR=data/train/UltraFeedback2
LLMS="[ \
    'Yi-1.5-34B-Chat', \
    'internlm2_5-20b-chat', \
    'Phi-3-medium-4k-instruct', \
    'Llama-3.1-8B-Instruct', \
    'Qwen2.5-Coder-7B-Instruct', \
]"

python src/compute_rewards.py \
    --data_dir $DATA_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --engine huggingface \
    --llms "${LLMS}" \
    --chunk_size 64
