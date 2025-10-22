set -e
set -o pipefail


DATA_DIR=data
TEST_SETS="['AlpacaEval', 'ArenaHard', 'MT-Bench', 'GSM8k', 'MATH', 'HumanEval', 'MBPP']"
CANDIDATE_LLMS="['Yi-1.5-34B-Chat', 'internlm2_5-20b-chat', 'Phi-3-medium-4k-instruct', 'Llama-3.1-8B-Instruct', 'Qwen2.5-Coder-7B-Instruct']"
OUTPUT_DIR="output/baselines"

ROUTER_TYPE=oracle
python src/eval_baseline.py \
    --router_type $ROUTER_TYPE \
    --candidate_llms "$CANDIDATE_LLMS" \
    --data_dir $DATA_DIR \
    --test_datasets "$TEST_SETS" \
    --output_dir $OUTPUT_DIR/$ROUTER_TYPE


ROUTER_TYPE=fixed
for i in 0 1 2 3 4; do
    python src/eval_baseline.py \
        --router_type $ROUTER_TYPE \
        --candidate_llms "$CANDIDATE_LLMS" \
        --target_index $i \
        --data_dir $DATA_DIR \
        --test_datasets "$TEST_SETS" \
        --output_dir $OUTPUT_DIR/$ROUTER_TYPE/$i &
done

ROUTER_TYPE=random
for i in 1 2 3; do
    OUTPUT_DIR="output/${VERSION}/${ROUTER_TYPE}/${i}"
    python src/eval.py \
        --router_type $ROUTER_TYPE \
        --candidate_llms "$CANDIDATE_LLMS" \
        --datasets "['Skywork-Reward-Gemma-2-27B-v0.2']" \
        --data_dir data/train/UltraFeedback \
        --output_dir $OUTPUT_DIR/valid \
        --eval_ids_path data/train/UltraFeedback/eval_ids.json &
    python src/eval.py \
        --router_type $ROUTER_TYPE \
        --candidate_llms "$CANDIDATE_LLMS" \
        --datasets $DATASETS \
        --data_dir data/test \
        --output_dir $OUTPUT_DIR/test &
done

wait
