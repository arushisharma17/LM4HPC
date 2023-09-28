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
module load pytorch/2.0.1

huggingface-cli whoami
python LM4HPC/Evaluation/evaluation.py --mcqa_dataset_file "mcq-single-orig.csv" --open_ended_dataset_file "text.csv" --model_names HuggingFaceH4/starchat-alpha gpt-4 gpt-3.5-turbo --output_csv "semantic_similarity_results_all.csv"
