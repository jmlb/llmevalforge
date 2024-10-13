from tqdm import tqdm
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate

from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import ChatPromptTemplate


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
    prompt_template = ChatPromptTemplate.from_messages([("system", "{system_prompt}"), ("user", "{instruction}")])
    chain_llm = prompt_template | llm

    for ix, record in enumerate(tqdm(dataset)):
        system_prompt = record["system_prompt"]
        user_query = record["instruction"]
        out = chain_llm.invoke({"system_prompt": system_prompt, "instruction": user_query})
            
        record["response_candidate_model"] = out
        dataset[ix] = record

    return dataset
