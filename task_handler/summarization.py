from tqdm import tqdm
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate

from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import ChatPromptTemplate


def run_summarization_task(llm: Any, dataset: List[Dict[str, Any]], **kwargs):
    """
    Executes a summarization task on a given dataset using a specified language model.

    Args:
        llm (Any): The model under assessment, responsible for generating summarization outputs.
        dataset (List[Dict[str, Any]]): A dataset where each record is a dictionary containing task-specific fields.
        **kwargs: Additional configuration parameters.

    Returns:
        List[Dict[str, Any]]: The updated dataset with model responses.
    """
    prompt_template = ChatPromptTemplate.from_messages([("system", "{system_prompt}"), ("user", "{instruction}")])
    chain_llm = prompt_template | llm

    for ix, record in enumerate(tqdm(dataset)):
        system_prompt = record["system_prompt"]
        user_query = record["instruction"]
        out = chain_llm.invoke({"system_prompt": system_prompt, "instruction": user_query})
            
        record["response_candidate_model"] = out
        dataset[ix] = record

    return dataset
        #except Exception as error:
        #    print(f"An error occurred while processing record: {error}")
        #    return record

    def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_record, record) for record in batch]
            return [future.result() for future in as_completed(futures)]

    updated_dataset = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[i:i+batch_size]
        updated_dataset.extend(process_batch(batch))

    return updated_dataset


# def run_summarization_task(llm, dataset):
#     """
#     Executes a summarization task on a given dataset using a specified language model and system prompt.

#     Args:
#         llm (object): The model under assessment, responsible for generating summarization outputs.
#         dataset (list of dict): A dataset where each record is a dictionary containing an "instruction" field for the user query
#                                 and a "system_prompt" field for the initial prompt given to the model.
#         system_prompt (str): A system-level prompt that provides the initial context or setup for the task.

#     Process:
#         - Iterates over each record in the dataset.
#         - Extracts the user query ("instruction") and system prompt.
#         - Constructs a prompt using `ChatPromptTemplate` with both the system prompt and user instruction.
#         - Feeds the constructed prompt to the model to generate the response.
#         - Stores the model's output (summarization response) in the field `response_candidate_model` of each record.

#     Returns:
#         list of dict: The updated dataset with an additional field `response_candidate_model` containing the model's generated summarization response for each record.
    
#     Raises:
#         KeyError: If a record is missing the "instruction" or "system_prompt" field.
#         Exception: For any other model invocation or runtime errors.
#     """
#     for ix, record in tqdm(enumerate(dataset), desc="Running Summarization task"):
#         try:
#             # Ensure that necessary fields exist in the record
#             if "instruction" not in record or "system_prompt" not in record:
#                 raise KeyError(f"Record at index {ix} is missing 'instruction' or 'system_prompt' fields.")
            
#             user_query = dedent(record["instruction"])
#             system_prompt = dedent(record["system_prompt"])

#             # Constructing the prompt
#             prompt = ChatPromptTemplate.from_messages([("system", system_prompt), 
#                                                        ("user", "{instruction}")])
#             # Attempt to invoke the model
#             chain_llm = prompt | llm
#             try:
#                 out = chain_llm.invoke(user_query)
#             except Exception as model_error:
#                 raise Exception(f"Model invocation failed at index {ix} with error: {model_error}")
            
#             # Save the model's output to the dataset
#             record["response_candidate_model"] = out
#             dataset[ix] = record

#         except KeyError as key_error:
#             print(f"Error in record at index {ix}: {key_error}")
#             continue  # Skip the current record and continue processing

#         except Exception as error:
#             print(f"An error occurred while processing record at index {ix}: {error}")
#             continue  # Skip the current record and continue processing
    
    
#     return dataset
