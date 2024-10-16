#!/bin/bash
##SBATCH --job-name=batch_job_benchmark_vllm
##SBATCH --nodes=1
##SBATCH --time=01:00:00
##SBATCH --constraint=gpu
##SBATCH --gpus=4
##SBATCH --account=m2404
##SBATCH --output=batch_job_%j.out

module load python/3.10
module load cudatoolkit/12.2

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"

python3 offline_inference_arxiv.py > offline_inference_arxiv.log
