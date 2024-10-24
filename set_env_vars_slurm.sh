#!/bin/bash
# select_gpu_device wrapper script
export ROCR_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=fork
exec $*
