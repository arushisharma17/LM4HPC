from prompts import EVALUATION_PROMPTS
from datasets import load_dataset
from lm4hpc.hpcpipeline import hpcpipelines
import argparse
import csv
import os

def load_model_return_response(model_name, prompt):
    """
    Load a question-answering model specified by `model_name` and return the model's response to a given `prompt`.

    This function initializes a pipeline for the 'openmp_question_answering' task using the specified model.
    It then processes a prompt and returns the generated response.

    Parameters:
    - model_name (str): The name of the pre-trained model to be loaded for the pipeline.
    - prompt (str): The input prompt or question to be passed to the model for generating a response.

    Returns:
    - response (dict): A dictionary containing the model's response. The structure of the response dictionary
                       is determined by the underlying model and the task.
    """

    try:
        # Initialize the LM4HPC pipeline with the specified model.
        OMP_QA_model = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")  
        # Generate response using the model.
        response = OMP_QA_model(prompt)

    except Exception as e:
        # Handle exceptions that may occur during model loading or prompt processing.
        print(f"An error occurred: {e}")
        response = None
    
    return response


def load_dataset_from_hub(dataset_type, data_file, test_mode):
    """
    Load a dataset from the Hugging Face Hub based on the dataset type and file provided.
    
    Parameters:
    - dataset_type (str): The type of dataset to load. Accepted values are "mcq" for multiple-choice questions
                          and "open_ended" for open-ended questions.
    - data_file (str): The filename or path of the dataset file to load. This file should be located in the 
                       Hugging Face Hub repository specified by the dataset name.
                       
                       For "mcq" type, expected files could be 'mcq-single.csv', 'rodinia-basic.csv', or 
                       'rodinia-advanced.csv'.
                       
                       For "open_ended" type, expected files could be 'text.csv' or 'code.csv'.
    - test_mode (bool): If True, only load the first two examples from the dataset for testing purposes.
                        
    Returns:
    - dataset (DatasetDict): A `datasets.DatasetDict` object containing the loaded dataset.
    
    Raises:
    - ValueError: If the dataset_type is not one of the accepted values.
    
    Example usage:
    mcq_dataset = load_dataset_from_hub("mcq", "mcq-single.csv")
    open_ended_dataset = load_dataset_from_hub("open_ended", "text.csv")
    """

    # Load MCQ-type dataset from the HPCPerfOpt-MCQA repository on the Hugging Face Hub.
    if dataset_type == "mcq":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files=data_file)

    # Load open-ended dataset from the HPCPerfOpt-Open-ended repository on the Hugging Face Hub.
    elif dataset_type == "open_ended":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-Open-ended", data_files=data_file)

    # Raise an error if an invalid dataset_type is provided.
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")
    
   # If test_mode is True, only take the first two examples from each split.
    if test_mode:
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(2))

    return dataset


def create_LLM_prompt_from_example(example,dataset_type, prompt_type):
    '''
    Generate a language model prompt for a given example, which can be multiple-choice or open-ended.

    Parameters:
    - example (dict): A dictionary containing the example data.
    - prompt_type (str): The type of prompt to be generated. Expected values are 'standard', 'cot', 'text', 'code', 'none'.

    Returns:
    - str: A formatted string that serves as a prompt for a language model, containing the instruction (if provided),
           and the necessary information extracted from the example.

    Raises:
    - KeyError: If expected keys are missing in the `example` dictionary for the respective prompt type.
    - ValueError: If the `prompt_type` is not one of the expected values.
    '''
    # If prompt_type is 'none', set instruction to an empty string
    instruction = ''
        
    if prompt_type != 'none':
        if dataset_type == 'mcq':
            try:
                instruction = EVALUATION_PROMPTS["MCQA"][prompt_type]
            except KeyError:
                raise ValueError(f"Prompt type '{prompt_type}' does not exist for dataset type 'MCQA' in EVALUATION_PROMPTS.")

            question = example['Code Snippet'] + " " + example['Question']
            options = f"A. {example['A']} B. {example['B']} C. {example['C']} D. {example['D']}"
            prompt = f"{instruction} {question} Options: {options}"

        elif dataset_type == 'open_ended':
            try:
                instruction = EVALUATION_PROMPTS["OPEN_ENDED"][prompt_type]
            except KeyError:
                raise ValueError(f"Prompt type '{prompt_type}' does not exist for dataset type 'OPEN_ENDED' in EVALUATION_PROMPTS.")
            question = example['Code Snippet'] + " " + example['Question']
            prompt = f"{instruction} {question}"

        else:
            raise ValueError(f"Invalid dataset type. Expected 'MCQA' or 'OPEN_ENDED', but got '{dataset_type}'.")


    return prompt



#Note: prompts DO NOT depend on evaluation type here. We evaluate the different types of prompts


def exact_match_evaluation(mcqa_dataset, model_name, args):
    '''Evaluate model based on exact match for MCQs'''
    correct = 0
    total = 0
    results = []

    for idx, example in enumerate(mcqa_dataset['train']):
        print(f"Example #{idx + 1}: {example}\n{'-' * 80}")
        prompt = create_LLM_prompt_from_example(example, args.dataset_type, args.prompt_type)

        if prompt is not None:
            print(f"Prompt #{idx + 1}:\n{prompt}\n{'-' * 80}")
            response = load_model_return_response(model_name, prompt)

            if response:  # Check if response is not empty
                print(response)
                correct_answer = example.get('Answer')  # Safely get 'Answer' key
                if correct_answer is not None:
                    is_correct = (correct_answer.strip() == response[0].strip())  # Strip spaces before comparison
                    if is_correct:
                        correct += 1
                    # Now we collect the results after each example is processed
                    result_dict = {
                        "prompt": prompt,
                        "response": response[0],
                        "correct_answer": correct_answer,
                        "is_correct": is_correct
                    }
                    results.append(result_dict)
                else:
                    print("Error: 'Answer' key is missing in the example.")
            else:
                print("Response is empty or None.")
        else:
            print("Prompt could not be generated due to missing data.")

        total += 1  # Increment total inside the loop

    accuracy = correct / total if total > 0 else 0  # Prevent division by zero
    return accuracy, results

#create separate csv file for each combo and store in one model folder. Later mix and match in bash script. 
def create_output_filename(dataset_type, data_file, prompt_type, eval_metric):
    '''
    Create an output filename based on dataset type, data file, prompt type, and evaluation metric.

    Parameters:
    - dataset_type (str): The type of the dataset (e.g., 'training', 'validation', 'test').
    - data_file (str): The path to the original data file.
    - prompt_type (str): The type of the prompt (e.g., 'multiple_choice', 'true_false').
    - eval_metric (str): The evaluation metric used (e.g., 'accuracy', 'f1_score').

    Returns:
    - str: A formatted output filename.
    '''

    # Extract the base name of the data file without extension
    base_name = os.path.splitext(os.path.basename(data_file))[0]

    # Create the output filename
    output_filename = f"{dataset_type}_{base_name}_{prompt_type}_{eval_metric}.csv"

    return output_filename

    '''output filename based on dataset type, prompt type and evaluation metric'''

def main(args):
    # Your code to load the dataset and evaluate the model would go here
    print(f"Loading {args.dataset_type} dataset from file: {args.data_file}")
    print(f"Using prompt type: {args.prompt_type}")
    print(f"Evaluating model(s): {', '.join(args.model_names)}")
    
     #Build the name of the output_csv based on the provided arguments- it should append results of all models.It should not override previous results of a specific model but it should append if tehy don't exist. 
    output_csv = f"{args.dataset_type}_{args.prompt_type}_{args.eval_type}_evaluation_results.csv"

    #Should iterate over prompt types and deal with models separately as they are likely to be more problematic.
    #need separate evaluation for each prompt type and each dataset type and for each model.
    #add models to a dictionary/config file as well. 

    dataset = load_dataset_from_hub(args.dataset_type, args.data_file, args.test_mode)
    print(f"Loaded dataset with {len(dataset['train'])} examples for evaluation.",dataset) 
    
    #need a loop for model_names
    for model_name in args. model_names:
        accuracy, results = exact_match_evaluation(dataset, model_name, args)
        store_exact_match_results_in_csv(results, output_csv)
        print(f"Model: {model_name} - Accuracy: {accuracy}")
        #exact_match_evaluation(dataset,model_name, args):

    '''
    for idx, example in enumerate(mcqa_dataset['train']):
        print(f"Example #{idx + 1}: {example}\n{'-' * 80}")
        #Create prompt based on example and prompt type
        prompt = create_LLM_prompt_from_example(example, args.prompt_type)
        # Check if the prompt is None, which could be due to an error.
        if prompt is not None:
            print(f"Prompt #{idx + 1}:\n{prompt}\n{'-' * 80}")
        else:
            print(f"Prompt #{idx + 1} could not be generated due to missing data.\n{'-' * 80}")
        #Get response from selected model
        response = load_model_return_response(model_name, prompt)
        print(response)

  #create prompt based on selection from EVALUATION_PROMPTS and relevant fileds from the dataset
  #loop on the dataset here and have the function do it for one example at a time

        exact_match_evaluation(example['Answer'],response)
    '''



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate a model using a dataset from the Hugging Face Hub.')

    # Add the arguments
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode to load a smaller part of the dataset.')

    parser.add_argument('--dataset_type',
                        type=str,
                        required=True,
                        choices=['mcq', 'open_ended'],
                        help='The type of dataset to load. '
                             'Use "mcq" for multiple choice questions or "open_ended" for open-ended questions.')

    parser.add_argument('--data_file',
                        type=str,
                        required=True,
                        help='The filename of the dataset file to load from the Hugging Face Hub repository. '
                             'For "mcq" type, possible values include "mcq-single.csv", "rodinia-basic.csv", or "rodinia-advanced.csv". '
                             'For "open_ended" type, possible values include "text.csv" or "code.csv".')

    parser.add_argument('--model_names',
                        type=str,
                        nargs='+',
                        required=True,
                        help='The name(s) of the model(s) to use for evaluation. Multiple model names can be provided separated by spaces.')

    parser.add_argument('--prompt_type',
                        type=str,
                        default='none',  # Set the default value to 'none'
                        choices=['standard', 'cot', 'text', 'code', 'none'],
                        help='The type of prompt to be used for the LLM. "none" will use no additional prompt information.')
    parser.add_argument('--eval_type',
                        type=str,
                        default='none',  # Set the default value to 'none'
                        choices=['exact_match','semantic_similarity','bleu_score','codebertscore','LLM_as_a_judge'],
                        help='The type of evaluation to be performed. "none" will use no additional prompt information.')

   #Build the name of the output_csv based on the provided arguments- it should append results of all models.It should not override previous results of a specific model but it should append if tehy don't exist. For each model type, for each prompt type,for each data_file-> append to same csv file. Maybe move this to bash script later. 
 #output_csv = f"{args.dataset_type}_{args.prompt_type}_{args.eval_type}_evaluation_results.csv"


    args = parser.parse_args()
    main(args)
