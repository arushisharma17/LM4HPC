import os
from datasets import load_dataset
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import argparse
import csv
# Local imports
from lm4hpc.hpcpipeline import hpcpipelines
import evaluate

# Constants
OPENAI_API_KEY = "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"
HUGGINGFACEHUB_API_TOKEN = "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"

# Environment Variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Set up external services
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

def bleu_score_eval():
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    perplexity = evaluate.load("perplexity", module_type="metric")
    # results = perplexity.compute(predictions=predictions, model_id='gpt2')

def semantic_similarity_eval(open_ended_dataset,model_name, num_rows=2):
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
    accuracy = correct_count / num_rows * 100
    print(f"Accuracy on the first {num_rows} rows: {accuracy}%")
    
    # Return both accuracy and the list of results
    return correct_count / num_rows, results

def write_results_to_csv(results, model_name, file_name='semantic_similarity_results.csv'):
    '''Function to write the evaluation results to a CSV file'''
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write headers only if the file is new
        if not file_exists:
            headers = ["Model Name", "Question", "Response", "Correct Answer", "Cosine Similarity", "Is Correct"]
            csvwriter.writerow(headers)

        # Write results to CSV
        for result in results:
            csvwriter.writerow([model_name, result["question"], result["response"], result["correct_answer"], result["similarity"], result["is_correct"]])

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate models for semantic similarity.")
    parser.add_argument("--mcqa_dataset_file", default="mcq-single-orig.csv", help="Path to the MCQA dataset file.")
    parser.add_argument("--open_ended_dataset_file", default="code.csv", help="Path to the open-ended dataset file.")
    parser.add_argument("--model_names", nargs='+', default=["gpt-4", "gpt-3.5-turbo", "HuggingFaceH4/starchat-alpha" ], help="List of model names to evaluate.")
    parser.add_argument("--output_csv", default="semantic_similarity_results.csv", help="Name of the output CSV file.")

    # Parse the arguments
    args = parser.parse_args()

    # Load datasets using provided paths
    mcqa_dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files=args.mcqa_dataset_file)
    print(mcqa_dataset)

    open_ended_dataset = load_dataset("sharmaarushi17/HPCPerfOpt-Open-ended", data_files=args.open_ended_dataset_file)
    num_rows = open_ended_dataset['train'].num_rows
    print("num_rows", num_rows)
    print(open_ended_dataset)
    
    # Use provided list of models
    for model_name in args.model_names:
        print(f"Evaluating model: {model_name}")
        accuracy, results = semantic_similarity_eval(open_ended_dataset, model_name, num_rows)
        print(f"Accuracy on the first {num_rows} rows: {accuracy * 100:.2f}%")
        
        # Write results to CSV using provided name and model_name
        write_results_to_csv(results, model_name, args.output_csv)



