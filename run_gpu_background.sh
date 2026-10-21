#!/usr/bin/env bash
# ==============================================================================
# SpiroFusion-Net: Background GPU Execution Script (NVIDIA RTX 3090)
# ==============================================================================
GPU_ID=${1:-0}
EPOCHS=${2:-50}
BATCH_SIZE=${3:-128}

echo "==========================================================================="
echo "  Starting SpiroFusion-Net A-to-Z Pipeline in Background on GPU ${GPU_ID}"
echo "  Epochs: ${EPOCHS} | Batch Size: ${BATCH_SIZE} | Log: train.log"
echo "==========================================================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u run_pipeline.py \
    --gpu 0 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --resume > train.log 2>&1 &

PID=$!
echo ${PID} > train.pid
echo "[✓] Background process launched with PID: ${PID}"
echo "[*] To monitor live progress, run: tail -f train.log"
