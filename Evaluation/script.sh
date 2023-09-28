#!/bin/bash
#SBATCH -A m2956_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 5:00:00
#SBATCH -N 1
#SBATCH -c 32
nvidia-smi
export HF_HOME=/pscratch/sd/s/sharma21/hf/
cd $SCRATCH
source lm4hpcenv/bin/activate
echo "OPENAI_API_KEY='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'" >> ~/.bashrc
module load pytorch/2.0.1

huggingface-cli whoami
python LM4HPC/Evaluation/evaluation.py
