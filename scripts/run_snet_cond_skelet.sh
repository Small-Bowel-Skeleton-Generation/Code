#!/bin/bash

# ==============================================================================
# Script for training and generating with the snet model.
#
# This script is designed to be run from the root of the project directory.
#
# Usage:
#   bash scripts/run_snet_cond_skelet.sh <mode> <stage_flag> <category> [gpu_ids]
#
# Arguments:
#   <mode>:         train | generate
#   <stage_flag>:   lr | hr (for high/low resolution training stages)
#   <category>:     e.g., skeleton7_rifle, skeleton7
#   [gpu_ids]:      Comma-separated GPU IDs (e.g., 0,1,2). Defaults to '0'.
#
# Examples:
#   bash scripts/run_snet_cond_skelet.sh train hr skeleton7_rifle 0,1,2
#   bash scripts/run_snet_cond_skelet.sh train lr skeleton7 0
#   bash scripts/run_snet_cond_skelet.sh generate hr skeleton7_rifle 0
# ==============================================================================

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

# --- Script Arguments ---
MODE=${1}
STAGE_FLAG=${2}
CATEGORY=${3}
GPU_IDS=${4:-"0"} # Default to GPU 0 if not provided

# --- Paths Configuration ---
# !!! IMPORTANT: Please update these paths to match your environment !!!
# PYTHON_EXEC="python" # Assumes 'python' is in your PATH and points to the correct environment
PYTHON_EXEC="/home/data/anaconda3/envs/lzc_octfusion/bin/python" # Assumes 'python' is in your PATH and points to the correct environment

# Set logs_dir based on mode and stage_flag
# LR training -> logs/skeleton_diff_lr
# HR generation -> logs/skeleton_diff_hr
if [ "$MODE" = "train" ] && [ "$STAGE_FLAG" = "lr" ]; then
    LOGS_DIR="/home/data/liangzhichao/Code/Tree-diffuison-update/logs/skeleton_diff_lr"
elif [ "$MODE" = "generate" ] && [ "$STAGE_FLAG" = "hr" ]; then
    LOGS_DIR="/home/data/liangzhichao/Code/Tree-diffuison-update/logs/skeleton_diff_hr"
else
    LOGS_DIR="logs"  # default fallback
fi

# Base directories for data and pretrained models.
# These are example paths, please change them to your actual paths.
DATA_BASE_DIR="/mnt/gemlab_data_2/User_database/liangzhichao"
CODE_BASE_DIR="/home/data/liangzhichao/Code/octfusion-main"

# Conditional data directory
COND_DIR_TRAIN="${DATA_BASE_DIR}/simulated_mask_data_8"
COND_DIR_GENERATE="${DATA_BASE_DIR}/QC_mask_mat"

# Checkpoint paths
VQ_CKPT="/home/data/liangzhichao/Code/Tree-diffuison-update/logs/skeleton_vae/ckpt/vae_steps-latest.pth"
PRETRAIN_CKPT="/home/data/liangzhichao/Code/Tree-diffuison-update/logs/skeleton_diff_lr/ckpt/df_steps-latest.pth"
CKPT_GENERATE="/home/data/liangzhichao/Code/Tree-diffuison-update/logs/skeleton_diff_lr/ckpt/df_steps-latest.pth"


# --- Model Configuration ---
MODEL='union_2t'
DATASET_MODE='snet'
NOTE="test"
COND=True
VQ_MODEL="GraphVAE"

DF_YAML="octfusion_${DATASET_MODE}_cond.yaml"
DF_CFG="configs/${DF_YAML}"
VQ_YAML="vae_${DATASET_MODE}_eval_skeleton.yaml"
VQ_CFG="configs/${VQ_YAML}"

# --- Hyperparameter Configuration ---
LR=2e-4
MIN_LR=1e-6
UPDATE_LEARNING_RATE=0
WARMUP_EPOCHS=40
EMA_RATE=0.999
CKPT_NUM=3
SEED=42
DEBUG=0

# Set epochs and batch size based on stage flag
if [ "$STAGE_FLAG" = "lr" ]; then
    EPOCHS=3000
    BATCH_SIZE=128
else # hr
    EPOCHS=500
    BATCH_SIZE=2
fi

# --- Display & Log Configuration ---
DISPLAY_FREQ=5000
PRINT_FREQ=25
SAVE_STEPS_FREQ=5000
SAVE_LATEST_FREQ=500

# --- Sanity Checks ---
if [ -z "$MODE" ] || [ -z "$STAGE_FLAG" ] || [ -z "$CATEGORY" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: bash $0 <mode> <stage_flag> <category> [gpu_ids]"
    exit 1
fi

# --- Execution Logic ---
echo "Starting script..."
echo "Mode: ${MODE}, Stage: ${STAGE_FLAG}, Category: ${CATEGORY}, GPUs: ${GPU_IDS}"

# Set conditional directory
if [ "$MODE" = "generate" ]; then
    COND_DIR="${COND_DIR_GENERATE}"
else
    COND_DIR="${COND_DIR_TRAIN}"
fi

# Set experiment name
# For HR generate, flatten to logs_dir directly by using name='.'
if [ "$MODE" = "generate" ] && [ "$STAGE_FLAG" = "hr" ]; then
    NAME="."
else
    NAME="${CATEGORY}_union/${MODEL}_${NOTE}_lr${LR}"
fi

# Base command
CMD="train.py \
    --name ${NAME} \
    --logs_dir ${LOGS_DIR} \
    --gpu_ids ${GPU_IDS} \
    --mode ${MODE} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --min_lr ${MIN_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --update_learning_rate ${UPDATE_LEARNING_RATE} \
    --ema_rate ${EMA_RATE} \
    --seed ${SEED} \
    --model ${MODEL} \
    --stage_flag ${STAGE_FLAG} \
    --df_cfg ${DF_CFG} \
    --ckpt_num ${CKPT_NUM} \
    --category ${CATEGORY} \
    --vq_model ${VQ_MODEL} \
    --vq_cfg ${VQ_CFG} \
    --vq_ckpt ${VQ_CKPT} \
    --display_freq ${DISPLAY_FREQ} \
    --print_freq ${PRINT_FREQ} \
    --save_steps_freq ${SAVE_STEPS_FREQ} \
    --save_latest_freq ${SAVE_LATEST_FREQ} \
    --debug ${DEBUG} \
    --cond ${COND} \
    --cond_dir ${COND_DIR}"

# Add checkpoints based on mode
if [ "$MODE" = "train" ]; then
    if [ -n "$PRETRAIN_CKPT" ]; then
        CMD="${CMD} --pretrain_ckpt ${PRETRAIN_CKPT}"
        echo "Training with pretrain_ckpt: ${PRETRAIN_CKPT}"
    fi
else # generate or other modes
    if [ -n "$CKPT_GENERATE" ]; then
        CMD="${CMD} --ckpt ${CKPT_GENERATE}"
        echo "Running with ckpt: ${CKPT_GENERATE}"
    fi
fi

# Ensure generate mode does not trigger train-mode directory setup
if [ "$MODE" = "generate" ]; then
    CMD="${CMD} --isTrain False"
fi

# --- GPU and Launch Configuration ---
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "[*] Starting on $(hostname), GPU(s): ${GPU_IDS}"
echo "[*] Command to be executed:"

if [ ${NUM_GPUS} -gt 1 ]; then
    HOST_NODE_ADDR="localhost:27000"
    LAUNCH_CMD="--nnodes=1 --nproc_per_node=${NUM_GPUS} --rdzv-backend=c10d --rdzv-endpoint=${HOST_NODE_ADDR}"
    
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON_EXEC} -m torch.distributed.launch ${LAUNCH_CMD} ${CMD}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON_EXEC} -m torch.distributed.launch ${LAUNCH_CMD} ${CMD}
else
    # For single GPU, we can also use torch.distributed.launch for consistency
    # if the training script is designed for it.
    LAUNCH_CMD="--nnodes=1 --nproc_per_node=1"
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON_EXEC} -m torch.distributed.launch ${LAUNCH_CMD} ${CMD}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON_EXEC} -m torch.distributed.launch ${LAUNCH_CMD} ${CMD}
fi

echo "Script finished."
