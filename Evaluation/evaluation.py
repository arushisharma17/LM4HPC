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

def bleu_score_eval():
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    perplexity = evaluate.load("perplexity", module_type="metric")
    # results = perplexity.compute(predictions=predictions, model_id='gpt2')
    
def exact_match_evaluation(mcqa_dataset,model_name):
    '''Adapted from DoctorGPT exact match method for MCQ evaluation--removed conversation history'''

    correct = 0
    total = 0
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")

    results = []

    for idx, example in enumerate(mcqa_dataset['train']):
        question = example['startphrase (context + question)']

        try:
            options = "A." + example['ending0'] + "B." + example['ending1'] + "C." + example['ending2'] + "D." + example['ending3']
            input_sample = question + options + "Please output the correct option A,B,C or D only"
            response = OMP_QA_sc(input_sample)
            generated_text = response
            correct_answer = example['Answer']
            predicted_answer_idx = generated_text[0]

            is_correct = (correct_answer == predicted_answer_idx)
            if is_correct:
                correct += 1

            results.append((model_name, question, generated_text, correct_answer, is_correct))
            total += 1

        except TypeError:
            print(f"Found TypeError (probably due to None value) in line {idx + 1}. Endings: {example['ending0'], example['ending1'], example['ending2'], example['ending3']}")

    return correct / total, results

def semantic_similarity_eval(open_ended_dataset,model_name):
    '''Adapted from DoctorGPT exact match method for MCQ evaluation--removed conversation history'''
    print("entered semantic similarity function")
    correct_count = 0

    # Initialize sentence transformer model
    #free embeddings!
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("embedder")
    OMP_QA_sc=hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files = "", langchain_embedding ="")
    predictions=[]
    references=[]
    results = []  # This list will store the results for each question

    for example in open_ended_dataset['train']:
        input_sample = example['startphrase (Context + Question)']  # Replace 'text_field' with the actual field name
        print("input_sample", input_sample)
        response = OMP_QA_sc(input_sample)
        print("response",response)
        answer = example['Answer']

        # Generate embeddings for the response and correct answer
        response_embedding = embedder.encode(response, convert_to_tensor=True)
        #print("response embedding", response_embedding)
        correct_answer_embedding = embedder.encode(answer, convert_to_tensor=True)

        #print("answer embedding",correct_answer_embedding)

        # Compute cosine similarity
        cosine_similarity = util.pytorch_cos_sim(response_embedding, correct_answer_embedding).item()
        print('the similarity is ' + str(cosine_similarity))
        is_correct = cosine_similarity >= 0.3 # Adjust the threshold as needed, >30% threshold

        if is_correct:
            correct_count += 1

        print(f"Correct Answer: {answer}")
        print(f"Is Model's Response Correct? {is_correct}\n")

        results.append({
            "question": input_sample,
            "response": response,
            "correct_answer": answer,
            "similarity": cosine_similarity,
            "is_correct": is_correct
        })
    
    # Print the accuracy
    num_rows = open_ended_dataset['train'].num_rows
    print("num_rows", num_rows)
    accuracy = correct_count / num_rows * 100
    print(f"Accuracy on the first {num_rows} rows: {accuracy}%")

    # Return both accuracy and the list of results
    return correct_count / num_rows, results



def evaluate_models(models, evaluation_type, file_name, dataset_type, args):
    # Determine dataset files based on dataset_type
    if dataset_type == "open_ended":
        dataset_files = args.open_ended_dataset_file
    elif dataset_type == "mcqa":
        dataset_files = args.mcqa_dataset_file
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    # Loop through dataset files
    for data_file in dataset_files:
        print(f"Loading data from: {data_file}")
        
        # Load the dataset
        if dataset_type == "open_ended":
            dataset = load_dataset("sharmaarushi17/HPCPerfOpt-Open-ended", data_files=data_file)
            num_rows = dataset['train'].num_rows
            print("num_rows", num_rows)
            print(dataset)
        elif dataset_type == "mcqa":
            dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files=data_file)
            print(dataset)

        # Now perform evaluations
        for model_name in models:
            print(f"Evaluating model: {model_name}")

            # Determine which evaluation method to use
            if evaluation_type == "semantic_similarity":
                accuracy, results = semantic_similarity_eval(dataset, model_name)
                headers = ["Model Name", "Dataset", "Question", "Response", "Correct Answer", "Cosine Similarity", "Is Correct"]
            elif evaluation_type == "exact_match":
                accuracy, results = exact_match_evaluation(dataset, model_name)
                headers = ['Model Name', 'Dataset', 'Question', 'Response', 'Correct Answer', 'Is Correct']
            else:
                raise ValueError(f"Invalid evaluation_type: {evaluation_type}")

            print(f"Accuracy for {model_name}: {accuracy * 100:.2f}%")

            # Write results to CSV
            file_exists = os.path.isfile(file_name)
            with open(file_name, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                # Write headers only if the file is new
                if not file_exists:
                    csvwriter.writerow(headers)

                if evaluation_type == "semantic_similarity":
                    for result in results:
                        csvwriter.writerow([model_name, data_file, result["question"], result["response"], result["correct_answer"], result["similarity"], result["is_correct"]])
                else:
                    for result in results:
                        csvwriter.writerow([model_name, data_file, result["question"], result["response"], result["correct_answer"], result["is_correct"]])

# Assuming you've parsed args using argparse before calling this function.
# Example Usage:
# evaluate_models(["gpt-4", "gpt-3.5-turbo"], "semantic_similarity", "semantic_similarity_results.csv", "open_ended", args)
# evaluate_models(["gpt-4", "gpt-3.5-turbo"], "exact_match", "evaluation_results_rodinia.csv", "mcqa", args)

import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import argparse
import csv

# Local imports
from lm4hpc.hpcpipeline import hpcpipelines
import sys

# Set up external services
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

def evaluate_models(models, evaluation_type, file_name, dataset_type, args):
    # Fetch dataset files using helper function
    dataset_files = get_dataset_files(dataset_type, args)

    # Loop through dataset files
    for data_file in dataset_files:
        print(f"Loading data from: {data_file}")

        # Load dataset data using helper function
        dataset = load_dataset_data(dataset_type, data_file)

        # Now perform evaluations
        for model_name in models:
            print(f"Evaluating model: {model_name}")

            # Determine which evaluation method to use
            if evaluation_type == "semantic_similarity":
                accuracy, results = semantic_similarity_eval(dataset, model_name)
                headers = ["Model Name", "Dataset", "Question", "Response", "Correct Answer", "Cosine Similarity", "Is Correct"]
            elif evaluation_type == "exact_match":
                accuracy, results = exact_match_evaluation(dataset, model_name)
                headers = ['Model Name', 'Dataset', 'Question', 'Response', 'Correct Answer', 'Is Correct']
            else:
                raise ValueError(f"Invalid evaluation_type: {evaluation_type}")

            print(f"Accuracy for {model_name}: {accuracy * 100:.2f}%")

            # Write results to CSV using helper function
            write_results_to_csv(results, headers, model_name, data_file, file_name)


def get_dataset_files(dataset_type, args):
    """Determine the dataset files based on the dataset type."""
    if dataset_type == "open_ended":
        return args.open_ended_dataset_file
    elif dataset_type == "mcqa":
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

def write_results_to_csv(results, headers, model_name, data_file, file_name):
    """Write the evaluation results to a CSV file."""
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write headers only if the file is new
        if not file_exists:
            csvwriter.writerow(headers)

        if headers[-1] == "Is Correct":  # This is a simple check to determine the type of evaluation
            for result in results:
                csvwriter.writerow([model_name, data_file, result["question"], result["response"], result["correct_answer"], result["is_correct"]])
        else:
            for result in results:
                csvwriter.writerow([model_name, data_file, result["question"], result["response"], result["correct_answer"], result["similarity"], result["is_correct"]])


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate models for semantic similarity and exact match.")

    # Multiple dataset files can be input as a list separated by commas
    parser.add_argument("--mcqa_dataset_file", nargs='+', default=["mcq-single-orig.csv", "rodinia-chatgpt-mcq-orig.csv"], help="Paths to the MCQA dataset files.")
    parser.add_argument("--open_ended_dataset_file", nargs='+', default=["code.csv", "text.csv"], help="Paths to the open-ended dataset files.")
    parser.add_argument("--model_names", nargs='+', default=["HuggingFaceH4/starchat-alpha","gpt-4", "gpt-3.5-turbo"], help="List of model names to evaluate.")
    parser.add_argument("--semantic_similarity_output_csv", default="semantic_similarity_results.csv", help="Name of the output CSV file for semantic similarity evaluation.")
    parser.add_argument("--exact_match_output_csv", default="evaluation_results_rodinia.csv", help="Name of the output CSV file for exact match evaluation.")

    # Parse the arguments
    args = parser.parse_args()

    # Print the OpenAI API Key for debugging (consider removing this for security reasons!)
    print(openai.api_key)

    # Run evaluations
    evaluate_models(args.model_names, "semantic_similarity", args.semantic_similarity_output_csv, "open_ended", args)
    evaluate_models(args.model_names, "exact_match", args.exact_match_output_csv, "mcqa", args)

