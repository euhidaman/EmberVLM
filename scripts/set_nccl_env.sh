#!/bin/bash
# NCCL Environment Variables for Stable Distributed Training

# Increase timeout to 30 minutes (from default 10 minutes)
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# Network optimizations
export NCCL_SOCKET_IFNAME=^lo,docker0  # Exclude loopback
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_IB_HCA=mlx5  # Mellanox IB adapter
export NCCL_NET_GDR_LEVEL=5  # Enable GPU Direct RDMA

# Debugging (set to 1 for verbose logs)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,ENV

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0  # Keep async for performance

# PyTorch distributed settings
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Set to OFF for production

echo "NCCL environment variables set for stable distributed training"
echo "Timeout: 1800 seconds (30 minutes)"

