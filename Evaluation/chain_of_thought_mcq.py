import os
from datasets import load_dataset
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import argparse
import csv
# Local imports
from lm4hpc.hpcpipeline import hpcpipelines
import evaluate
import sys

# Set up external services
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
print(openai.api_key)

def generate_dataset(dataset):
    # Initialize model pipeline
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")

    results = []

    # Define various types of prompts
    prompts = [
        "Given the above code snippet, can you generate OpenMP performance optimization multiple choice questions with a single correct answer?",
        "Based on the provided code, can you ask a yes/no question?",
        "Considering the code snippet above, can you create an open-ended question about its performance or structure?"
    ]

    for code_snippet in dataset:
        input_sample = code_snippet['']  # name of column containing code snippet in selected dataset
        
        # Replace newlines with \n newline character
        formatted_code = input_sample.replace("\n", "\\n")

        for prompt_template in prompts:
            full_prompt = formatted_code + prompt_template
            response = OMP_QA_sc(full_prompt)

            results.append({
                "prompt": full_prompt,
                "response": response,
            })

    # Write results to a CSV file
    with open("generated_questions.csv", "w", newline="") as csvfile:
        fieldnames = ["prompt", "response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models for semantic similarity and exact match.")
    parser.add_argument("--mcqa_dataset_file", nargs='+', default=["mcq-single-orig.csv"], help="Paths to the MCQA dataset files.")
    parser.add_argument("--open_ended_dataset_file", nargs='+', default=["code.csv"], help="Paths to the open-ended dataset files.")
    parser.add_argument("--model_names", nargs='+', default=["gpt-4", "gpt-3.5-turbo"], help="List of model names to evaluate.")
    args = parser.parse_args()

