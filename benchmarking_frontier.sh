#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -A csc569
#SBATCH -N 1
#SBATCH -J vllm_bench
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out
#SBATCH -C nvme

#module reset
#module load PrgEnv-gnu
#module load rocm/6.2.0.lua

conda init
conda activate vllm-bench

SCRATCH="/lustre/orion/scratch/prajwal/csc569/"

export PYTHONPATH="/opt/rocm-6.2.0/share/amd_smi/"

export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CRAY_MPICH_ROOTDIR}/gtl/lib"

## this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lustre/orion/scratch/prajwal/csc547/aws-ofi-rccl/build/lib"
#export NCCL_DEBUG=INFO
export FI_CXI_ATS=0

export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_CROSS_NIC=1
export NCCL_NET_GDR=3
#
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

export HF_HOME=${HF_HOME:-"$SCRATCH/hf_cache"}
export TRANSFORMERS_HOME=${TRANSFORMERS_HOME:-"$SCRATCH/hf_cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$SCRATCH/hf_cache"}

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


BATCH_SIZE=$1
PROMPT_LENGTH=$2
TP=$3

# This is executing the ping_ping_gpu_aware.exe in the container with `apptainer exec`.
# The program passes data allocated on GPU memory of increasing size back and forth between
# two MPI processes across two nodes.
# The --rocm flag is required to support AMD GPUs inside the container.
#srun  -N 1  --tasks-per-node 1 --gpus-per-task=4 apptainer exec --bind /lustre/orion/csc569/scratch/prajwal/vllm-mnt:/mnt/vllm-mnt --writable-tmpfs --workdir `pwd` --rocm test.sif ./set_env_vars_slurm.sh python examples/offline_inference.py
srun -N 1 --tasks-per-node 1 --gpus-per-task=8 ./set_env_vars_slurm.sh python pssg-benchmarking/offline_inference_arxiv.py -c pssg-benchmarking/configs_profile/offline_inference_arxiv_${BATCH_SIZE}_${TP}.ini -pl ${PROMPT_LENGTH} -o ./vllm_benchmarking_Llama31_8B_bs_${BATCH_SIZE}_pl_${PROMPT_LENGTH}_tp_${TP}.csv
