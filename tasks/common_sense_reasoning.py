import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from lib.utils import load_dataset


logger = logging.getLogger(__name__)


class Common_Sense_ReasoningTask:
    """
    The Common Sense Reasoning Task is designed to evaluate a language model's ability to 
    handle vague and ambiguous customer queries in an e-commerce context. The focus is on understanding 
    abstract or incomplete descriptions, extracting key insights, and applying general principles to 
    generate relevant and helpful recommendations.

    This class provides functionality to run evaluation tasks by processing
    input test cases or datasets, generating model responses, and preparing
    results for evaluation. It supports input from either a YAML dataset file
    or a list of test cases.
    """
    
    def __init__(self, model: Any):
        """Initialize with a language model."""
        self.model = model
    
    def run_task(self, dataset_path_or_cases: str | List[dict]) -> List[Dict[str, Any]]:
        """
        Run Common Sense reasoning task with the candidate model
        
        Args:
            dataset_path_or_cases: Path to dataset YAML file or a list of test cases
            
        Returns:
            List of results with model responses
        """
        # Check if dataset_path_or_cases is a string (file path) or a list of test cases
        if isinstance(dataset_path_or_cases, str):
            # Load dataset from file
            try:
                dataset = load_dataset(dataset_path_or_cases)["test_cases"]
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise
        elif isinstance(dataset_path_or_cases, list):
            # Use the provided list of test cases
            dataset = dataset_path_or_cases
        else:
            raise ValueError("dataset_path_or_cases must be a string or a list of dictionaries")

            
        # Validate dataset
        required_fields = ["instruction", "system_prompt"]
        for test_case in dataset:
            missing = [f for f in required_fields if f not in test_case]
            if missing:
                raise ValueError(f"Missing required fields in cases: {missing}")
        
        # Setup prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", "{instruction}")
        ])
        chain = prompt | self.model
        
        # Process each test case
        results = []
        for test_case in tqdm(dataset, desc="Running Common Sense Reasoning task"):
            try:
                # Get model response
                response = chain.invoke({
                    "system_prompt": test_case["system_prompt"],
                    "instruction": test_case["instruction"]
                })
                
                # Store result
                result = {
                    **test_case,  # Keep original fields
                    "model_response": response.content if hasattr(response, 'content') else response,
                    "score": None,  # Will be filled by evaluator
                    "feedback": None  # Will be filled by evaluator
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing case {test_case.get('id', '?')}: {e}")
                result = {
                    **test_case,
                    "model_response": f"ERROR: {str(e)}",
                    "score": None,
                    "feedback": None
                }
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage for testing
    import json
    import yaml
    from models import create_model
    
    # Create test dataset
    test_cases = [{
        "case_id": 1,
        "instruction": "what cloth do I need when it is cold outside",
        "expected_response": "a thick jacket",
        "system_prompt": "You are a helpful assistant.",
        "challenges": ""
    }]
    
    with open("dummy_dataset.yaml", "w") as f:
        yaml.dump({"metadata": "", "test_cases": test_cases}, f)
    
    # Run test
    model = create_model(model_name="gpt-3.5-turbo")
    runner = Common_Sense_ReasoningTask(model)
    results = runner.run_task("dummy_dataset.yaml")
    
    # Print results
    print(json.dumps(results, indent=2))
