#!/bin/bash

# ==============================================================================
# Script for training and running inference with the VAE model.
#
# This script is designed to be run from the root of the project directory.
#
# Usage:
#   bash scripts/run_snet_vae_skelet.sh <mode> <category> [gpu_ids]
#
# Arguments:
#   <mode>:         train | generate | inference_vae
#   <category>:     e.g., skeleton7
#   [gpu_ids]:      Comma-separated GPU IDs (e.g., 0,1,2). Defaults to '0'.
#
# Examples:
#   bash scripts/run_snet_vae_skelet.sh train skeleton7 0,1,2
# ==============================================================================

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

# --- Script Arguments ---
MODE=${1}
CATEGORY=${2}
GPU_IDS=${3:-"0"} # Default to GPU 0 if not provided

# --- Paths Configuration ---
# !!! IMPORTANT: Please update these paths to match your environment !!!
PYTHON_EXEC="python" # Assumes 'python' is in your PATH
LOGS_DIR="logs"
CODE_BASE_DIR="/home/data/liangzhichao/Code/octfusion-main" # Example path

# Checkpoint path for VQ model (used in generation)
VQ_CKPT="${CODE_BASE_DIR}/logs/skeleton7_union/test_snet_lr1e-3/ckpt/vae_steps-latest.pth"

# --- Model Configuration ---
MODEL='vae'
DATASET_MODE='snet'
NOTE="test"
VQ_MODEL="GraphVAE"

# Config files
DF_YAML="octfusion_${DATASET_MODE}_cond.yaml"
VQ_YAML="vae_${DATASET_MODE}_train_skeleton.yaml"
DF_CFG="configs/${DF_YAML}"
VQ_CFG="configs/${VQ_YAML}"

# --- Hyperparameter Configuration ---
LR=1e-3
MIN_LR=1e-6
UPDATE_LEARNING_RATE=0
WARMUP_EPOCHS=40
EPOCHS=300
BATCH_SIZE=64
EMA_RATE=0.999
CKPT_NUM=3
SEED=42
DEBUG=0

# --- Display & Log Configuration ---
DISPLAY_FREQ=1000
PRINT_FREQ=25
SAVE_STEPS_FREQ=3000
SAVE_LATEST_FREQ=500

# --- Sanity Checks ---
if [ -z "$MODE" ] || [ -z "$CATEGORY" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: bash $0 <mode> <category> [gpu_ids]"
    exit 1
fi

# --- Execution Logic ---
echo "Starting script..."
echo "Mode: ${MODE}, Category: ${CATEGORY}, GPUs: ${GPU_IDS}"

# Set experiment name
NAME="${CATEGORY}_union/${NOTE}_${DATASET_MODE}_lr${LR}"

# For generation/inference, use the config files from the log directory
if [ "$MODE" = "generate" ] || [ "$MODE" = "inference_vae" ]; then
    DF_CFG="${LOGS_DIR}/${NAME}/${DF_YAML}"
    VQ_CFG="${LOGS_DIR}/${NAME}/${VQ_YAML}"
    CKPT="${LOGS_DIR}/${NAME}/ckpt/df_steps-latest.pth"
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
    --df_cfg ${DF_CFG} \
    --ckpt_num ${CKPT_NUM} \
    --category ${CATEGORY} \
    --vq_model ${VQ_MODEL} \
    --vq_cfg ${VQ_CFG} \
    --display_freq ${DISPLAY_FREQ} \
    --print_freq ${PRINT_FREQ} \
    --save_steps_freq ${SAVE_STEPS_FREQ} \
    --save_latest_freq ${SAVE_LATEST_FREQ} \
    --debug ${DEBUG}"

# Add checkpoint if it exists (for continuing training or generation)
if [ -n "$CKPT" ]; then
    CMD="${CMD} --ckpt ${CKPT}"
    echo "Running with ckpt: ${CKPT}"
fi

# Add VQ checkpoint specifically for generation mode
if [ "$MODE" = "generate" ]; then
    CMD="${CMD} --vq_ckpt ${VQ_CKPT}"
    echo "Using VQ ckpt for generation: ${VQ_CKPT}"
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
    LAUNCH_CMD="--nnodes=1 --nproc_per_node=1"

    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON_EXEC} -m torch.distributed.launch ${LAUNCH_CMD} ${CMD}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} ${PYTHON_EXEC} -m torch.distributed.launch ${LAUNCH_CMD} ${CMD}
fi

echo "Script finished."
