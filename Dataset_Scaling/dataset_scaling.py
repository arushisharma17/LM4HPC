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
from datasets import load_dataset

# Set up external services
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
print(openai.api_key)


def generate_dataset(dataset, model_name, output_csv_base):

    # Initialize model pipeline
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")

    # Define various types of prompts as dictionaries with an index and text
    prompts = [
            {"index": 0, "text": "Generate 2 OpenMP performance optimization multiple choice questions based on the given code snippet. The generated questions should be in json format with the following fields: Question :<generated question>, Options:<Four options A, B,C and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C' or 'D'>"},
            {"index": 1, "text": "Generate 2 OpenMP performance optimization multiple choice questions based on the given code snippet. The generated questions should be in json format with the following fields: Question :<generated question>, Options:<Four options A, B,C and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C' or 'D'>"},           
            {"index": 2, "text": "Generate 2 OpenMP performance optimization Yes/No questions based on the given code snippet. The questions should be in json format with the following fields: Question: <generated question> Answer: <Correct answer to the generated question 'Yes' or 'No'>"},

            {"index": 3, "text": "Generate 2 OpenMP performance optimization Yes/No questions based on the given code snippet. The questions should be in json format with the following fields: Question: <generated question> Answer: <Correct answer to the generated question 'Yes' or 'No'>"},

            {"index":4, "text": "Generate 2 open-ended OpenMP performance optimization questions based on the given code snippet. The questions should be in json format with the following fields: Question: <generated question> Answer: <Correct answer to the generated question>"}
    ]

    # Selected Rodinia samples
    row_indices = [2, 11]

    for prompt in prompts:

        # Construct the CSV filename for this prompt
        output_csv = f"Outputs/{output_csv_base}_prompt_{prompt['index']}.csv"

        # Open the CSV file for writing
        with open(output_csv, "w", newline="") as csvfile:
            fieldnames = ["code_snippet", "prompt", "response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx in row_indices:
                code_snippet = dataset['train']['Code (optional)'][idx]
                print(code_snippet)
                full_prompt = code_snippet + " " + prompt["text"]
                print("full prompt", full_prompt)
                response = OMP_QA_sc(full_prompt)
                print(response)

                writer.writerow({
                    "code_snippet": code_snippet,
                    "prompt": full_prompt,
                    "response": response
                })

    print("CSV generation completed.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models for semantic similarity and exact match.")
    parser.add_argument("--mcqa_dataset_file", nargs='+', default=["mcq-single-orig.csv"], help="Paths to the MCQA dataset files.")
    parser.add_argument("--open_ended_dataset_file", nargs='+', default=["code.csv"], help="Paths to the open-ended dataset files.")
    parser.add_argument("--model_names", nargs='+', default=["gpt-4"], help="List of model names to evaluate.")
    args = parser.parse_args()
   
    
    #Load Rodinia Dataset
    rodinia_dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files="rodinia-chatgpt-mcq-orig.csv")
    print(rodinia_dataset)

    #generate_dataset(rodinia_dataset,"databricks/dolly-v2-3b","rodinia-generated-questions.csv")
    generate_dataset(rodinia_dataset,"gpt-4","rodinia-generated-questions")

    
    #Load ompify dataset
    #ompify_dataset




