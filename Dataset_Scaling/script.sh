#!/bin/bash
#SBATCH -A m2956_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH -c 32
nvidia-smi
export HF_HOME=/pscratch/sd/s/sharma21/hf/
cd $SCRATCH
export OPENAI_API_KEY='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'
source ~/.bashrc
source lm4hpcenv/bin/activate
module load pytorch/2.0.1

huggingface-cli whoami
python LM4HPC/Dataset_Scaling/dataset_scaling.py --model_names gpt-4
