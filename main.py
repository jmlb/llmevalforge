import argparse
import logging
from pathlib import Path

from lib.utils import load_config_files
from models import create_model
import importlib
from typing import Any
from evaluator import Evaluator
from lib.utils import load_dataset, save_dataset


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_task_executor(module_path: str, class_name: str) -> Any:
    """
    Dynamically import a class from a module.
    
    Args:
        module_path: Path to the module (e.g., 'tasks.summarization')
        class_name: Name of the class to import
        
    Returns:
        The imported class
        
    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class doesn't exist in module
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class from the module
        return getattr(module, class_name)
        
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in module {module_path}: {e}")
    

def run_evaluation(config_dir: Path, output_dir: Path, verbose: bool = False):
    """Run the evaluation pipeline."""
    tasks_cfg_fname = "tasks.yaml"
    cand_model_cfg_fname = "candidate_model.yaml"
    eval_model_cfg_fname = "evaluator.yaml"

    try:
        # Setup
        setup_logging(verbose)
        configs, error_msgs = load_config_files(config_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create models
        logger.info("Creating models...")
        model_config = configs[cand_model_cfg_fname]
        model = create_model(
            model_name=model_config["model"]["name"],
            model_type=model_config["model"]["type"],
            **model_config["parameters"]
        )

        # evaluatorModel
        evaluator_config = configs[eval_model_cfg_fname]
        evaluator = create_model(
            model_name=evaluator_config["model"]["name"],
            model_type=evaluator_config["model"]["type"],
            api_key_source=evaluator_config["model"]["api_key_source"],
            **evaluator_config["parameters"]
        )
        
        evaluator = Evaluator(evaluator, evaluator_config["evaluator_prompt"])

        # Run each task
        for task_name, task_config in configs[tasks_cfg_fname]["tasks"].items():
            try:
                logger.info(f"Running task: {task_name}")
                # Create task runner and evaluator
                ExecutorClass = load_task_executor(module_path=task_config["import_lib"], 
                                                   class_name=task_config["executor"])
                task_runner = ExecutorClass(model)

                dataset_fname = task_config["dataset_path"]
                dataset = load_dataset(dataset_fname)

                # Run task
                results = task_runner.run_task(
                    dataset_path_or_cases=dataset["test_cases"]
                )
                
                # Run evaluation if required
                if task_config["run_evaluation"]:
                    results = evaluator.evaluate_results(
                        results,
                        output_path=None
                    )
                
                logger.info(f"Completed task: {task_name}")

                output_fname = output_dir / f"{task_name}_results.yaml"
                logger.info(f"Save results in: {output_fname}")
                save_dataset(path_to_fname=output_fname, 
                             dataset={"metadata": dataset["metadata"], "test_cases": results})
                
            except Exception as e:
                logger.error(f"Failed to run task {task_name}: {e}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Tool")
    
    parser.add_argument(
        "--config",
        type=Path,
        default="config",
        help="Path to configuration directory"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    run_evaluation(args.config, args.output, args.verbose)


if __name__ == "__main__":
    main()
