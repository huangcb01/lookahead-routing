set -e
set -o pipefail


MODEL_PATH=01-ai/Yi-1.5-34B-Chat
DATA_PATH=data/train/UltraFeedback/instructions.jsonl
MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_PATH=data/train/UltraFeedback/responses/${MODEL_NAME}.jsonl
CACHE_DIR=cache/generate_responses/UltraFeedback/${MODEL_NAME}

python src/generate_responses.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --cache_dir $CACHE_DIR \
    --engine vllm \
    --chunk_size 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_new_tokens 4096 \
    --gpu_memory_utilization 0.86 \
    --max_num_seqs 64 \
    --preemption_mode recompute