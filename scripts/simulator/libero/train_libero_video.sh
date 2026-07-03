WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=${NGPUS:-8}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/datasets}
DATAPATH=${DATAPATH:-${DATA_ROOT}/processed_data/meta/libero_all_norm_aug.pkl}
ACTION_TOKENIZER_PATH=${ACTION_TOKENIZER_PATH:-${PROJECT_ROOT}/pretrain/fast}
PRETRAIN=${PRETRAIN:-${PROJECT_ROOT}/logs/ckpts/WORLD_MODEL_POSTTRAIN}
MODEL_CONFIG=${MODEL_CONFIG:-${PROJECT_ROOT}/configs/moe_fast_video.json}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/scripts/sft/zero3_offload.json}
EXP_NAME=${EXP_NAME:-UNIVLA_LIBERO_VIDEO_BS192_8k}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/logs/${EXP_NAME}}

MAX_STEPS=${MAX_STEPS:-8000}
SAVE_STEPS=${SAVE_STEPS:-2000}
LOGGING_STEPS=${LOGGING_STEPS:-20}
WARMUP_STEPS=${WARMUP_STEPS:-50}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-12}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-3}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
FRAMES=${FRAMES:-2}
ACTION_FRAMES=${ACTION_FRAMES:-10}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-3200}
LEARNING_RATE=${LEARNING_RATE:-8e-5}
MIN_LEARNING_RATE=${MIN_LEARNING_RATE:-5e-6}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}

export PYTHONPATH=$(pwd)

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train/train_moe.py \
    --model_name_or_path "${PRETRAIN}" \
    --model_config_path "${MODEL_CONFIG}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --learning_rate ${LEARNING_RATE} \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate ${MIN_LEARNING_RATE} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path "${DATAPATH}" \
    --max_steps ${MAX_STEPS} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps ${WARMUP_STEPS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --frames ${FRAMES} \
    --action_frames ${ACTION_FRAMES} \
    --max_position_embeddings ${MAX_POSITION_EMBEDDINGS} \
    --seed 42 \
    --logging_steps ${LOGGING_STEPS} \
    --gradient_checkpointing True \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --use_gripper True \
    --video_format "interleave" \
    --action_tokenizer_path "${ACTION_TOKENIZER_PATH}" \
    "$@"
