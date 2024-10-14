import os
import argparse
import importlib
from typing import Dict, Any
from dotenv import load_dotenv

from task_handler.utils import load_yaml, save_yaml_custom_sort


def create_model(model_config: Dict[str, Any]):
    """
    Instantiates and returns a model object based on the provided configuration.

    Args:
        model_config (Dict[str, Any]): A dictionary containing the model configuration.
            - 'module' (str): The module path where the model class is located.
            - 'class' (str): The name of the model class to instantiate.
            - 'params' (dict): A dictionary of parameters to pass to the model class constructor.

    Returns:
        object: An instance of the specified model class, initialized with the provided parameters.

    Raises:
        ImportError: If the specified module cannot be imported.
        AttributeError: If the specified class does not exist in the module.
        TypeError: If the parameters do not match the model class constructor.
    """
    model_class = getattr(importlib.import_module(model_config['module']), model_config['class'])
    return model_class(**model_config['params'])


def load_evaluator(evaluator_config: Dict[str, Any]):
    """
    Loads and returns the evaluation function from a specified module.

    This function dynamically imports a module specified in the evaluator configuration
    and retrieves the `run_evaluation` function from it. This allows for flexible
    integration of different evaluation strategies.

    Args:
        evaluator_config (Dict[str, Any]): A dictionary containing configuration details
            for the evaluator. It must include a "path" key that specifies the module
            path to import.

    Returns:
        Callable: The `run_evaluation` function from the specified module.

    Raises:
        ModuleNotFoundError: If the specified module cannot be found.
        AttributeError: If the `run_evaluation` function is not found in the module.
    """
    evaluator_module = importlib.import_module(evaluator_config["path"])
    return getattr(evaluator_module, "run_evaluation")


def load_task(task_name: str):
    """
    Dynamically loads and returns a task function from the specified task module.

    This function imports a module from the `task_handler` package based on the 
    provided task name and retrieves the corresponding task function. The task 
    function is expected to be named in the format `run_<task_name>_task`.

    Args:
        task_name : str
            The name of the task to load. This should correspond to a module within 
            the `task_handler` package and a function within that module.
    Returns:
        function
            The task function from the specified module, which can be executed to 
            perform the task.

    Raises:
        ModuleNotFoundError
            If the specified task module does not exist within the `task_handler` package.
        AttributeError
            If the specified task function does not exist within the task module.
    """
    task_module = importlib.import_module(f"task_handler.{task_name}")
    return getattr(task_module, f"run_{task_name}_task")


def run_task(task_name: str, model: Any, evaluator: Any, dataset: Dict[str, Any], task_config: Dict[str, Any], evaluator_config: Dict[str, Any]):
    """
    Executes a specified task using the provided model, evaluator, and dataset.

    Args:
        - task_name (str): The name of the task to be executed. This is used to load the appropriate task function.
        - model (Any): The model instance that will be used to perform the task.
        - evaluator (Any): The evaluator function or model used to assess the model's output.
        - dataset (Dict[str, Any]): The dataset on which the task will be performed. It should be structured according to the task's requirements.
        - task_config (Dict[str, Any]): Configuration settings specific to the task, including parameters that guide task execution.
        - evaluator_config (Dict[str, Any]): Configuration settings for the evaluator, which may include parameters for evaluation criteria and scoring.

    Returns:
        - Dict[str, Any]: The dataset with results from the task execution. If evaluation is required, it includes evaluation results.

    Notes:
        - The function first loads the task function using the task name and executes it with the model and dataset.
        - If the task does not require evaluation (as indicated by `task_config["requires_eval"]`), the function returns the dataset immediately.
        - If evaluation is required, the evaluator configuration is updated with task-specific settings, and the evaluator is run on the dataset.
    """
    task_func = load_task(task_name)
    dataset = task_func(model, dataset, **task_config)
    if not task_config["run_eval"]:
        return dataset
    
    evaluator_config.update(task_config)
    return evaluator(dataset, **evaluator_config)


def get_api_key(config: Dict[str, Any]) -> str:
    """
    Retrieve the API key based on the provided configuration.

    This function fetches the API key from a specified source as defined in the
    configuration dictionary. The source can either be an environment variable
    or a file. If the source is an environment variable, it retrieves the key
    from the 'OPENAI_API_KEY' environment variable. If the source is a file, it
    reads the key from the specified file path.

    Args:
        config : Dict[str, Any]
            A dictionary containing configuration settings. It should include:
            - 'api_key_source': A string indicating the source of the API key.
            Acceptable values are 'env' for environment variable and 'file' for
            file-based retrieval.
            - 'api_key_file': (Optional) A string specifying the file path to read
            the API key from, required if 'api_key_source' is 'file'.

    Returns:
        str
            The API key as a string.

    Raises:
        ValueError
            If the 'api_key_source' is neither 'env' nor 'file', a ValueError is
            raised indicating an unsupported API key source.
    """
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
    evaluator_prompts = load_yaml(configs["evaluator"]["prompt_file"])
    configs["evaluator"]["system_prompt"] = evaluator_prompts["system_prompt"]
    configs["evaluator"]["user_prompt"] = evaluator_prompts["user_prompt"]

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
        save_yaml_custom_sort(results, os.path.join(output_dir, f"{task_name}_results.yaml"))
        print(f"{task_name.capitalize()} task completed. Results saved to {task_name}_results.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run various tests on language models.")
    parser.add_argument("--config", default="configs.yaml", help="Path to the configuration file")
    parser.add_argument("--output", default="output", help="Directory to save result files")
    
    args = parser.parse_args()
    main(args.config, args.output)