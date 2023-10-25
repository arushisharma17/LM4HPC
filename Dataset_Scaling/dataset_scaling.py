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

def generate_dataset(dataset, model_name, output_csv):
    
    # Initialize model pipeline
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")

    # Define various types of prompts
    prompts = [
            "Generate 10 OpenMP performance optimization multiple choice questions based on the given code snippet? The generated questions should be in json format with fields Question :<generated question>, Answer: <Solution to the generated question A, B, C or D>", "Generate OpenMP performance optimization questions based on the provided code. The questions should be based on advanced OpenMP conceptts with an answer 'Yes' or 'No'?", "Considering the code snippet above, can you create an open-ended question about optimizing the code for best performance using OpenMP?"
    ]
    

    #Selected Rodinia samples
    #row_indices = [2, 11, 21,41,67,86,115,127]
    row_indices = [2,11]

    # Open the CSV file for writing
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["prompt", "response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx in row_indices:
            code_snippet = dataset['train']['Code (optional)'][idx]
            print(idx, code_snippet)
            input_sample = code_snippet  # name of column containing code snippet in selected dataset
            
            # Replace newlines with \n newline character
            #formatted_code = input_sample.replace("\n", "\\n")

            for prompt_template in prompts:
                print(prompt_template)
                full_prompt = input_sample + " " + prompt_template
                response = OMP_QA_sc(full_prompt)
                print(response)

                # Write the result immediately to the CSV
                writer.writerow({
                    "prompt": full_prompt,
                    "response": response,
                })

    # Return the results for further processing or verification
#    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models for semantic similarity and exact match.")
    parser.add_argument("--mcqa_dataset_file", nargs='+', default=["mcq-single-orig.csv"], help="Paths to the MCQA dataset files.")
    parser.add_argument("--open_ended_dataset_file", nargs='+', default=["code.csv"], help="Paths to the open-ended dataset files.")
    parser.add_argument("--model_names", nargs='+', default=["gpt-4", "gpt-3.5-turbo"], help="List of model names to evaluate.")
    args = parser.parse_args()
   
    
    #Load Rodinia Dataset
    rodinia_dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files="rodinia-chatgpt-mcq-orig.csv")
    print(rodinia_dataset)

    #generate_dataset(rodinia_dataset,"databricks/dolly-v2-3b","rodinia-generated-questions.csv")
    generate_dataset(rodinia_dataset,"gpt-4","rodinia-generated-questions1.csv")

    
    #Load ompify dataset
    #ompify_dataset




