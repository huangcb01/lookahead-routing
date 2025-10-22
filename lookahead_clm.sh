set -e
set -o pipefail


MODEL_PATH=HuggingFaceTB/SmolLM2-135M
DATA_DIR=data
TRAIN_SETS="UltraFeedback/Skywork-Reward-Gemma-2-27B-v0.2 OpenMathInstruct2_filtered2/gt2-Skywork-Reward-Gemma-2-27B-v0.2 SelfOSSInstructSC2_filtered2/responses"
TEST_SETS="AlpacaEval ArenaHard MT-Bench GSM8k MATH HumanEval MBPP"
TRAIN_SLICE="(0,1)"
CANDIDATE_LLMS="Yi-1.5-34B-Chat internlm2_5-20b-chat Phi-3-medium-4k-instruct Llama-3.1-8B-Instruct Qwen2.5-Coder-7B-Instruct"
SCORE_NORM=min-max
MODEL_ID_TYPE=special

LOSS_TYPE=BCE
TEMPERATURE=0.1      # For KL Loss
BCE_THRESHOLD=0.8    # For BCE Loss
SFT_LOSS_WEIGHT=0.5

BATCH_SIZE=2
ACCUMULATION_STEPS=32
LEARNING_RATE=1e-5
LR_SCHEDULER_TYPE=cosine
EPOCHS=2
EVAL_STEPS=50
MAX_LENGTH=2048
ADDITIONAL_ARGS="--use_liger"
ADDITIONAL_ARGS="$ADDITIONAL_ARGS --attn_implementation flash_attention_2"
# ADDITIONAL_ARGS="$ADDITIONAL_ARGS --gradient_checkpointing"
# ADDITIONAL_ARGS="$ADDITIONAL_ARGS --deepspeed config/deepspeed/ds_z3_config.json"
# ADDITIONAL_ARGS="$ADDITIONAL_ARGS --resume_from_checkpoint last-checkpoint"

model_name=$(basename $MODEL_PATH)
cudas=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
total_batch_size=$((BATCH_SIZE * ACCUMULATION_STEPS * cudas))
output_dir="output/${TRAIN_SETS//\//-}/${TRAIN_SLICE}_${SCORE_NORM}_${MAX_LENGTH}_${MODEL_ID_TYPE}/${model_name}/Causal-${LOSS_TYPE}_SFTLoss-${SFT_LOSS_WEIGHT}_Temp-${TEMPERATURE}_BCEThreshold-${BCE_THRESHOLD}/LR-${LR_SCHEDULER_TYPE}-${LEARNING_RATE}_BS-${total_batch_size}_EP-${EPOCHS}"
output_dir=$(echo "$output_dir" | sed 's/ /-/g')
available_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
launcher=$([ "$cudas" -gt 1 ] && echo "torchrun --nproc_per_node $cudas --master_port $available_port" || echo "python")

# =================== DEBUG =======================
# ACCUMULATION_STEPS=1
# ADDITIONAL_ARGS="$ADDITIONAL_ARGS --max_train_samples 500"
# EVAL_STEPS=20
# output_dir=output/tmp
# launcher="python -m debugpy --connect localhost:50971 --wait-for-client"
# =================================================

mkdir -p $output_dir
cp "$0" $output_dir/train.sh

echo "Training..."
$launcher \
    src/train.py causal \
    --seed 42 \
    --backbone_name_or_path $MODEL_PATH \
    --data_dir "$DATA_DIR" \
    --train_datasets $TRAIN_SETS \
    --test_datasets $TEST_SETS \
    --train_slice $TRAIN_SLICE \
    --candidate_llms $CANDIDATE_LLMS \
    --model_id_type $MODEL_ID_TYPE \
    --score_normalization $SCORE_NORM \
    --max_prompt_length $((MAX_LENGTH - 10)) \
    --max_length $MAX_LENGTH \
    --loss_type $LOSS_TYPE \
    --kl_temperature $TEMPERATURE \
    --bce_threshold $BCE_THRESHOLD \
    --sft_loss_weight $SFT_LOSS_WEIGHT \
    --use_logits_to_keep True \
    --dataset_num_proc 32 \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 4 \
    --output_dir $output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_steps $EVAL_STEPS \
    --save_total_limit 2 \
    --metric_for_best_model eval_reward \
    --load_best_model_at_end \
    --logging_steps 10 \
    --report_to none \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION_STEPS \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --num_train_epochs $EPOCHS \
    --fp16 \
    $ADDITIONAL_ARGS \
    $DEBUG \
    2>&1 | tee $output_dir/train.log
