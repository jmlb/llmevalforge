import os
import argparse
import importlib
from typing import Dict, Any
from dotenv import load_dotenv

from utils.utils import load_yaml, save_yaml


def create_model(model_config: Dict[str, Any]):
    """Create a model based on the configuration."""
    model_class = getattr(importlib.import_module(model_config['module']), model_config['class'])
    return model_class(**model_config['params'])


def load_evaluator(evaluator_config: Dict[str, Any]):
    """Load an evaluator based on the configuration."""
    evaluator_module = importlib.import_module("model_handler.evaluators")
    return getattr(evaluator_module, "run_evaluation")


def load_task(task_name: str):
    """Load a task function based on the task name."""
    task_module = importlib.import_module(f"task_handler.{task_name}")
    return getattr(task_module, f"run_{task_name}_task")


def run_task(task_name: str, model: Any, evaluator: Any, dataset: Dict[str, Any], task_config: Dict[str, Any], evaluator_config: Dict[str, Any]):
    """Run a specific task with the given model, evaluator, and dataset."""
    task_func = load_task(task_name)
    dataset = task_func(model, dataset, **task_config)
    evaluator_config.update(task_config)
    return evaluator(dataset, **evaluator_config)


def get_api_key(config: Dict[str, Any]) -> str:
    """Get the API key based on the configuration."""
    api_key_source = config.get('api_key_source', 'env')
    if api_key_source == 'env':
        return os.getenv('OPENAI_API_KEY')
    elif api_key_source == 'file':
        with open(config['api_key_file'], 'r') as f:
            return f.read().strip()
    else:
        raise ValueError(f"Unsupported API key source: {api_key_source}")


def main(config_path: str, output_dir: str):
    load_dotenv()  # Load environment variables from .env file if it exists
    configs = load_yaml(config_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    candidate_model = create_model(configs['candidate_model'])
    
    # Get the API key
    api_key = get_api_key(configs['evaluator'])
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or specify the api_key_file in the config.")
    
    # Add the API key to the evaluator config
    configs['evaluator']['model_params']['openai_api_key'] = api_key
    
    evaluator = load_evaluator(configs['evaluator'])
    evaluator_config = configs["evaluator"]

    for task_name, task_config in configs['evaluation_tasks'].items():
        dataset = load_yaml(task_config['dataset_file'])
        if not dataset:
            raise ValueError(f"The dataset for task {task_name} is empty")
        
        results = run_task(task_name, candidate_model, evaluator, dataset, task_config, evaluator_config)
        save_yaml(results, os.path.join(output_dir, f"{task_name}_results.yaml"))
        print(f"{task_name.capitalize()} task completed. Results saved to {task_name}_results.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run various tests on language models.")
    parser.add_argument("--config", default="configs.yaml", help="Path to the configuration file")
    parser.add_argument("--output", default="output", help="Directory to save result files")
    
    args = parser.parse_args()
    main(args.config, args.output)