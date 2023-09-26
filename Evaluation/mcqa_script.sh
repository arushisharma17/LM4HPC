#!/bin/bash
#SBATCH -A m2956_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 5:00:00
#SBATCH -N 1
#SBATCH -c 32
nvidia-smi
export HF_HOME=/pscratch/sd/s/sharma21/hf/
cd /global/homes/s/sharma21/
source lm4hpcenv/bin/activate
module load pytorch/2.0.1

huggingface-cli whoami
python LM4HPC/Evaluation/mcqa_eval.py --mcqa_dataset_file "mcq-single-orig.csv" --open_ended_dataset_file "text.csv" --model_names gpt-4 gpt-3.5-turbo HuggingFaceH4/starchat-alpha -output_csv "exact_match_results.csv"
