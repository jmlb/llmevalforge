from tqdm import tqdm
from textwrap import dedent
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from task_handler.utils import custom_sort, validate_test_dataset


SORTED_LEGACY_KEYS = ["case_id", 
               "category", 
               "sub_category", 
               "system_prompt", 
               "instruction",
               "expected_response",
               "potential_challenges",
               "difficulty_level"]
SORTED_NEW_KEYS = ["response_candidate_model"]
SORTED_EVAL_KEYS = ["score", "scorer_feedback"]

def run_general_knowledge_task(llm: Any, dataset: List[Dict[str, Any]], **kwargs):
    """
    Executes a summarization task on a given dataset using a specified language model.

    Args:
        llm (Any): The model under assessment, responsible for generating text outputs.
        dataset (List[Dict[str, Any]]): A dataset where each record is a dictionary containing task-specific fields.
        **kwargs: Additional configuration parameters.

    Returns:
        List[Dict[str, Any]]: The updated dataset with model responses.
    """
    validate_test_dataset(dataset, required_keys=SORTED_LEGACY_KEYS)
    prompt_template = ChatPromptTemplate.from_messages([("system", "{system_prompt}"), ("user", "{instruction}")])
    chain_llm = prompt_template | llm
    all_fields = SORTED_LEGACY_KEYS + SORTED_NEW_KEYS + SORTED_EVAL_KEYS

    for ix, record in enumerate(tqdm(dataset)):
        system_prompt = dedent(record["system_prompt"])
        user_query = dedent(record["instruction"])
        out = chain_llm.invoke({"system_prompt": system_prompt, "instruction": user_query})
            
        record["response_candidate_model"] = out
        for fld in SORTED_EVAL_KEYS:
            record[fld] = None

        dataset[ix] = custom_sort(record, all_fields)

    return dataset
