import time
from tqdm import tqdm
from textwrap import dedent
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from task_handler.utils import custom_sort, validate_test_dataset


# sorted list of fields in the original test case, a dict
SORTED_LEGACY_KEYS = ["case_id", 
                    "category", 
                    "sub_category", 
                    "system_prompt", 
                    "instruction", 
                    "expected_output_length", 
                    "expected_response", 
                    "difficulty_level"]
# sorted list of fields added to the case after inference with the candidate model
SORTED_NEW_KEYS = ["response_candidate_model",
                   "inference_time",
                   "output_word_count",
                   "words_per_second"]
# sorted list of fields added to the case after evaluation with judge-model
SORTED_EVAL_KEYS = ["score", "judge_feedback"]


def run_inference_speed_task(llm: Any, 
                             dataset: List[Dict[str, Any]], 
                             **kwargs
                             ):
    """
    Executes an inference speed task on a given dataset using 
    a specified language model.

    Args:
        llm (Any): The model under assessment, responsible 
        for generating outputs.
        dataset (List[Dict[str, Any]]): A dataset where each 
        record is a dictionary containing task-specific 
        fields.
        **kwargs: Additional configuration parameters.

    Returns:
        List[Dict[str, Any]]: The updated dataset with model
        responses and inference times.
    """
    validate_test_dataset(dataset, 
                          required_keys=SORTED_LEGACY_KEYS)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "{instruction}")
    ])
    all_fields = SORTED_LEGACY_KEYS +\
                 SORTED_NEW_KEYS +\
                 SORTED_EVAL_KEYS

    def process_record(
            record: Dict[str, Any]
            ) -> Dict[str, Any]:
        record["response_candidate_model"] = None
        record["inference_time"] = None
        record["output_word_count"] = None
        record["words_per_second"] = None
        try:
            system_prompt = dedent(record["system_prompt"])
            instruction = dedent(record["instruction"])
            target_length = record["expected_output_length"]

            chain_llm = prompt_template | llm
            # Measure inference time
            start_time = time.time()
            instruction_with_length = (f"""
                                       {instruction} 
                                       Generate approximately 
                                       {target_length} words.
                                       """).strip()
            out = chain_llm.invoke({
                "system_prompt": f"{system_prompt}", 
                "instruction": instruction_with_length})
            end_time = time.time()

            inference_time = end_time - start_time
            word_count = len(out.split())

            record["response_candidate_model"] = out
            record["inference_time"] = inference_time
            record["output_word_count"] = word_count
            wrd_per_sec = word_count / inference_time if inference_time > 0 else 0
            record["words_per_second"] = wrd_per_sec
        except Exception as error:
            print(f"An error occurred while processing record: {error}")

        for fld in SORTED_EVAL_KEYS:
            record[fld] = None

        return custom_sort(record, all_fields)

    for ix, record in tqdm(enumerate(dataset), 
                           desc="Processing inference speed tasks"):
        updated_record = process_record(record)
        dataset[ix] = updated_record

    return dataset