import streamlit as st
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob
import inspect
import importlib.util
from models import create_model
from evaluator import Evaluator
from main import load_task_executor

from lib.utils import load_config_files, get_available_tasks
from lib.automatic_evaluation_page import run_single_evaluation


CANDIDATE_CONFIG_FILE = "candidate_model.yaml"
EVALUATOR_CONFIG_FILE = "evaluator.yaml"
TASKS_CONFIG_FILE = "tasks.yaml"


def get_executor_class(task_file: str):
    """get the name of the main class in a task file."""
    task_class = None
    print(task_file)
    module = importlib.import_module(task_file)
    try:
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(
            task_file.stem, 
            str(task_file)
        )
        if spec is None or spec.loader is None:
            return "Could not load task file"
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get all classes from the module
        classes = inspect.getmembers(module, inspect.isclass)
        # Find the task class (assuming it's the only one or ends with 'Task')

        for name, cls in classes:
            if name.endswith('Task'):
                task_class = getattr(module, name)
        if not task_class:
            print(f"No class Executor ending with <BASENAME>Task coudl be foudn in {task_file}")
        
        return  task_class
    except Exception as e:
        return f"Error findign task class: {str(e)}"


def get_class_docstring(task_class_obj) -> str:
    """Extract docstring from the main class in a task file."""
    try:       
        return inspect.getdoc(task_class_obj) or "No documentation available"
    except Exception as e:
        return f"Error loading task documentation: {str(e)}"


def get_dataset_info(dataset_path: str) -> dict:
    """Get information about the dataset."""
    try:
        with open(dataset_path) as f:
            dataset = yaml.safe_load(f)
            if isinstance(dataset, dict):
                test_cases = dataset.get("dataset", [])
            else:  # If it's a list (old format)
                test_cases = dataset
            return {
                "num_cases": len(test_cases),
                "metadata": dataset.get("metadata", {}) if isinstance(dataset, dict) else {}
            }
    except Exception as e:
        return {"error": str(e)}


def get_datasets_for_task(task_type: str) -> List[str]:
    """Get all dataset files that match the task type."""
    datasets_dir = Path("datasets")
    datasets = []
    
    # Get all yaml files in the datasets directory and subdirectories
    for dataset_file in glob.glob(str(datasets_dir / "**/*.yaml"), recursive=True):
        try:
            with open(dataset_file) as f:
                dataset_config = yaml.safe_load(f)
                # Check if dataset matches task type
                if isinstance(dataset_config, dict) and \
                   dataset_config.get("metadata", {}).get("category", "").lower() == task_type.lower():
                    datasets.append(dataset_file)
        except Exception:
            continue
    
    return datasets
    
# def run_single_evaluation(
#     task_executor: Any,
#     model_config: dict,
#     evaluator_config: dict,
#     task_name: str,
#     test_case: List[dict],
#     output_dir: Optional[Path] = None
# ) -> List[Dict[str, Any]]:
#     """Run evaluation for a single test case or a list of test cases."""
#     try:
#         # Create model
#         try:
#             model = create_model(
#                 model_name=model_config["model"]["name"],
#                 model_type=model_config["model"]["type"],
#                 **model_config.get("parameters", {})
#             )
#         except Exception as e:
#             st.error(f"Failed to create model: {str(e)}")
#             return []

#         # Create evaluator
#         try:
#             eval_model = create_model(
#                 model_name=evaluator_config["model"]["name"],
#                 model_type=evaluator_config["model"]["type"],
#                 api_key_source=evaluator_config["model"]["api_key_source"],
#                 **evaluator_config.get("parameters", {})
#             )
#             evaluator = Evaluator(eval_model, evaluator_config["evaluator_prompt"])
#         except Exception as e:
#             st.error(f"Failed to create evaluator: {str(e)}")
#             return []

#         # Create task runner
#         task_runner = task_executor(model)

#         # Run task
#         results = task_runner.run_task(test_case)

#         # Run evaluation
#         evaluated_results = evaluator.evaluate_results(
#             results,
#             output_path=None if not isinstance(output_dir, str) else Path(output_dir)/f"{task_name}_results.json"
#         )

#         return evaluated_results

#     except Exception as e:
#         st.error(f"Evaluation failed: {str(e)}")
#         return []


def manual_evaluation_page():
    """Page for running manual evaluations."""


    st.title("Manual Evaluation")
    
    # Load configurations
    configs, msgs = load_config_files()
    for msg in msgs:
        st.error(msg)

    # Get configs
    # Get configs
    model_config = configs.get(CANDIDATE_CONFIG_FILE, {})
    if not model_config:
        st.error("Candidate Model config not found!")
    
    evaluator_config = configs.get(EVALUATOR_CONFIG_FILE, {})
    if not evaluator_config:
        st.error("Evaluator Model config not found!")
    
    tasks_config = configs.get(TASKS_CONFIG_FILE, {}).get("tasks", {})
    if not tasks_config:
        st.error("Tasks config not found!")

    # Configuration Overview Section
    with st.expander("Models Configuration Overview", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Candidate Model")
            st.json(model_config)
        
        with col2:
            st.subheader("Evaluator Model")
            st.json(evaluator_config)

        help="*Visit the configuration editor to edit the configuration for the candidate model and the evaluator."
        st.write(f"\n{help}")

    # Evaluation Settings Section
    st.header("Build a Test Case")
    
    # Task selection for selected entry
    selected_task = st.selectbox(
        "Select Task",
        options=get_available_tasks(),
        help="Choose which task to evaluate",
        key="selected_task_select"
    )

    if selected_task:
        # Load TaskExecutor: SummarizationTask, etc...
        TaskExecutor = load_task_executor(module_path=tasks_config[selected_task]["import_lib"], 
                                            class_name=tasks_config[selected_task]["executor"])
        doc = get_class_docstring(TaskExecutor)
        with st.expander("Task Description", expanded=True):
            st.markdown(doc)
        
        # Create form for manual test case entry
        with st.form("manual_test_case"):
            st.subheader("Test Case Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                system_prompt = st.text_area(
                    "System Prompt [Candidate Model]",
                    value="You are an expert at...",
                    help="System prompt for the candidate model",
                    height=100
                )
                
                expected_response = st.text_area(
                    "Expected Response",
                    value="The expected response from the candidate model...",
                    help="The expected response from the candidate model: the evaluator will compare the candidate actual reponse to the expected response.",
                    height=100
                )
            
            with col2:
                instruction = st.text_area(
                    "Instruction [Candidate Model]",
                    help="The task instruction or input",
                    height=100
                )
                
                potential_challenges = st.text_area(
                    "Potential Challenges",
                    help="Key aspects for the evaluator to focus on when assessing the candidate model's response.",
                    height=100
                )
            
            difficulty = st.selectbox(
                "Difficulty Level",
                options=["easy", "medium", "hard"]
            )
            
            submit_button = st.form_submit_button("Run Evaluation")
            
            if submit_button:
                try:
                    with st.spinner("Running evaluation..."):
                        # Create test case
                        test_case = {
                            "case_id": 1,
                            "category": selected_task,
                            "system_prompt": system_prompt,
                            "instruction": instruction,
                            "expected_response": expected_response,
                            "challenges": potential_challenges,
                            "difficulty_level": difficulty
                        }
                        
                        # Run evaluation
                        try:
                            results = run_single_evaluation(
                                task_executor=TaskExecutor,
                                model_config=model_config,
                                evaluator_config=evaluator_config,
                                task_name=selected_task,
                                test_case=[test_case],  # Wrap in list since API expects list
                                output_dir=None
                            )
                            
                            # Show results
                            st.success("Evaluation completed!")
                            st.subheader("Results")
                            
                            if results:
                                result = results[0]  # Get first (and only) result
                                st.write("**Response [Candidate Model]:**")
                                st.write(result.get('model_response', ''))
                                st.write("**Score [Evaluator Model]:** ", result.get('score', 0))
                                st.write("**Full response/Feedback [Evaluator Model]:**")
                                st.write(result.get('feedback', ''))
                                
                                # Show full result in expandable section
                                with st.expander("Full Result Details"):
                                    st.json(result)
                            
                        except Exception as e:
                            st.error(f"Error during evaluation: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error preparing test case: {str(e)}")