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

def bleu_score_eval():
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    perplexity = evaluate.load("perplexity", module_type="metric")
    # results = perplexity.compute(predictions=predictions, model_id='gpt2')
    

def exact_match_evaluation(mcqa_dataset, model_name):
    '''Evaluate model based on exact match for MCQs'''
    correct = 0
    total = 0
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")
    results = []
    instructions_std = "The following is a multiple choice question about openmp performance optimization. Output a single option from the four options as the final answer."
    instructions_cot = "The following is a multiple choice question about openmp performance optimization. Solve it in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer."
    instructions_list = [instructions_std, instructions_cot]
    for instruction in instructions_list:
        for idx, example in enumerate(mcqa_dataset['train']):
            question = example['startphrase( context + question)']
            try:
                options = f"A.{example['ending0']} B.{example['ending1']} C.{example['ending2']} D.{example['ending3']}"
                input_sample = f"{instruction} {question} {options}"
                response = OMP_QA_sc(input_sample)
                correct_answer = example['Answer']
                is_correct = (correct_answer == response[0])
                if is_correct:
                    correct += 1
                result_dict = {
                    "instruction": instruction,
                    "question": question,
                    "response": response[0], # assuming response[0] is the model's answer
                    "correct_answer": correct_answer,
                    "is_correct": is_correct
                }
                results.append(result_dict)
                total += 1
            except Exception as e:
                print(f"Error at line {idx + 1}. Error: {str(e)}")
    return correct / total, results

def semantic_similarity_eval(open_ended_dataset, model_name):
    '''Evaluate model based on semantic similarity'''

    correct_count = 0

    # Initialize sentence transformer model
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Initialize model pipeline
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")

    results = []

    for example in open_ended_dataset['train']:
        input_sample = example['startphrase (Context + Question)']
        response = OMP_QA_sc(input_sample)
        answer = example['Answer']

        # Generate embeddings for the response and correct answer
        response_embedding = embedder.encode(response, convert_to_tensor=True)
        correct_answer_embedding = embedder.encode(answer, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_similarity = util.pytorch_cos_sim(response_embedding, correct_answer_embedding).item()

        is_correct = cosine_similarity >= 0.3  # Adjust the threshold as needed

        if is_correct:
            correct_count += 1

        results.append({
            "question": input_sample,
            "response": response,
            "correct_answer": answer,
            "similarity": cosine_similarity,
            "is_correct": is_correct
        })

    # Using num_rows for the dataset length
    num_rows = open_ended_dataset['train'].num_rows
    return correct_count / num_rows, results


def evaluate_models(models, evaluation_type, file_name, dataset_type, args):
    # Fetch dataset files using helper function
    dataset_files = get_dataset_files(dataset_type, args)
    model_accuracies = {}

    # Loop through dataset files
    for data_file in dataset_files:
        print(f"Loading data from: {data_file}")

        # Load dataset data using helper function
        dataset = load_dataset_data(dataset_type, data_file)

        # Check if the CSV file exists to determine if headers need to be written
        file_exists = os.path.isfile(file_name)

        # Now perform evaluations
        for model_name in models:
            print(f"Evaluating model: {model_name}")

            # Determine which evaluation method to use
            if evaluation_type == "semantic_similarity":
                accuracy, results = semantic_similarity_eval(dataset, model_name)
                headers = ["Model Name", "Dataset", "Question", "Response", "Correct Answer", "Cosine Similarity", "Is Correct"]
            elif evaluation_type == "exact_match":
                accuracy, results = exact_match_evaluation(dataset, model_name)
                headers = ['Model Name', 'Dataset','Instruction', 'Question', 'Response', 'Correct Answer', 'Is Correct']
            else:
                raise ValueError(f"Invalid evaluation_type: {evaluation_type}")

            print(f"Accuracy for {model_name}: {accuracy * 100:.2f}%")
            
            # Storing accuracy of each model for final write
            model_accuracies[model_name] = accuracy

            # Write results to CSV immediately
            write_results_to_csv(results, headers, model_name, data_file, file_name, file_exists)
            
            # Update file_exists to True for subsequent writes
            file_exists = True
    
    # Write the final accuracies to the CSV file at the top
    with open(file_name, 'r') as original:
        data = original.read()
    
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model Name', 'Accuracy'])
        for model_name, accuracy in model_accuracies.items():
            writer.writerow([model_name, accuracy])
        writer.writerow([])  # Empty line
        csvfile.write(data)




def get_dataset_files(dataset_type, args):
    """Determine the dataset files based on the dataset type."""
    if dataset_type == "open_ended":
        print("Loading configuration", args.open_ended_dataset_file)
        return args.open_ended_dataset_file
    elif dataset_type == "mcqa":
        print("Loading configuration", args.mcqa_dataset_file)
        return args.mcqa_dataset_file
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

def load_dataset_data(dataset_type, data_file):
    """Load dataset based on dataset type and file."""
    if dataset_type == "open_ended":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-Open-ended", data_files=data_file)
        print("Number of rows:", dataset['train'].num_rows)
    elif dataset_type == "mcqa":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files=data_file)
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")
    print(dataset)
    return dataset

def write_results_to_csv(results, headers, model_name, data_file, file_name, file_exists):
    """Write the evaluation results to a CSV file."""
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(headers)
        for result in results:
            if "similarity" in result:
                csvwriter.writerow([model_name, data_file, result["question"], result["response"], result["correct_answer"], result["similarity"], result["is_correct"]])
            else:
                csvwriter.writerow([model_name, data_file, result["instruction"], result["question"], result["response"], result["correct_answer"], result["is_correct"]])



if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate models for semantic similarity and exact match.")

    # Multiple dataset files can be input as a list separated by commas
    parser.add_argument("--mcqa_dataset_file", nargs='+', default=["mcq-single-orig.csv", "rodinia-chatgpt-mcq-orig.csv"], help="Paths to the MCQA dataset files.")
    parser.add_argument("--open_ended_dataset_file", nargs='+', default=["code.csv", "text.csv"], help="Paths to the open-ended dataset files.")
    parser.add_argument("--model_names", nargs='+', default=["HuggingFaceH4/starchat-alpha","gpt-4", "gpt-3.5-turbo"], help="List of model names to evaluate.")
    parser.add_argument("--semantic_similarity_output_csv", default="LM4HPC/Evaluation/semantic_similarity_results.csv", help="Name of the output CSV file for semantic similarity evaluation.")
    parser.add_argument("--exact_match_output_csv", default="LM4HPC/Evaluation/evaluation_results_rodinia.csv", help="Name of the output CSV file for exact match evaluation.")

    # Parse the arguments
    args = parser.parse_args()

    # Print the OpenAI API Key for debugging (consider removing this for security reasons!)
    print(openai.api_key)

    # Run evaluations
    #evaluate_models(args.model_names, "semantic_similarity", args.semantic_similarity_output_csv, "open_ended", args)
    evaluate_models(args.model_names, "exact_match", args.exact_match_output_csv, "mcqa", args)

