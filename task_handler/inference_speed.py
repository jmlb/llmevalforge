import time
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate


def run_inference_speed_task(llm: Any, dataset: List[Dict[str, Any]], **kwargs):
    """
    Executes an inference speed task on a given dataset using a specified language model.

    Args:
        llm (Any): The model under assessment, responsible for generating outputs.
        dataset (List[Dict[str, Any]]): A dataset where each record is a dictionary containing task-specific fields.
        **kwargs: Additional configuration parameters.

    Returns:
        List[Dict[str, Any]]: The updated dataset with model responses and inference times.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a product description generator. Create concise product descriptions based on the given attributes."),
        ("human", "{instruction}")
    ])

    def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            instruction = record["instruction"]
            target_length = record["target_output_length"]

            chain_llm = prompt_template | llm

            # Measure inference time
            start_time = time.time()
            out = chain_llm.invoke({"instruction": f"{instruction} Generate approximately {target_length} words."})
            end_time = time.time()

            inference_time = end_time - start_time
            word_count = len(out.content.split())

            record["response_candidate_model"] = out.content
            record["inference_time"] = inference_time
            record["output_word_count"] = word_count
            record["words_per_second"] = word_count / inference_time if inference_time > 0 else 0

            return record
        except Exception as error:
            print(f"An error occurred while processing record: {error}")
            return record

    for ix, record in tqdm(enumerate(dataset), desc="Processing inference speed tasks"):
        updated_record = process_record(record)
        dataset[ix] = updated_record

    return dataset