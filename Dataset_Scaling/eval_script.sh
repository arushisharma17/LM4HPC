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
cd /pscratch/sd/s/sharma21/LM4HPC/Dataset_Scaling
python evals.py --dataset_type mcq --data_file mcq-single.csv --model_names gpt-4 --test_mode --prompt_type none

python evals.py --dataset_type mcq --data_file rodinia-basic.csv --model_names gpt-4

python evals.py --dataset_type mcq --data_file mcq-advanced.csv --model_names gpt-4
