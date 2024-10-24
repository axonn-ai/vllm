#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J vllm-benchmarking
#SBATCH --time=<time>
#SBATCH --mail-user=<user>@umd.edu
#SBATCH --mail-type=ALL
#SBATCH -A <account>

module load python/3.10
module load cudatoolkit/12.2

cd ..
source vllm_venv/bin/activate
cd pssg-benchmarking

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export HF_ACCESS_TOKEN="TOKEN_HERE"

python3 offline_inference_arxiv.py -c offline_inference_arxiv.ini &> offline_inference_arxiv.log
