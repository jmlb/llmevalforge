import time
import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
import glob
import inspect
from models import create_model
from evaluator import Evaluator

from main import load_task_executor
from lib.utils import load_config_files, get_available_tasks, load_dataset


CANDIDATE_CONFIG_FILE = "candidate_model.yaml"
EVALUATOR_CONFIG_FILE = "evaluator.yaml"
TASKS_CONFIG_FILE = "tasks.yaml"


def get_class_docstring(class_obj: Any) -> str:
    """Extract docstring from the main class in a task file."""
    return inspect.getdoc(class_obj) or "No documentation available"


def get_dataset_info(dataset_path: str) -> dict:
    """Get information about the dataset."""
    try:
        test_cases, metadata = load_dataset(dataset_path)
        return {
                "num_cases": len(test_cases),
                "metadata": metadata
            }
    except Exception as e:
        return {"error": str(e)}


def get_datasets_for_task(task_type: str) -> List[str]:
    """Get all dataset files that match the task type."""
    datasets_dir = Path("datasets")
    selected_datasets = []
    
    # Get all yaml files in the datasets directory and subdirectories
    print("list of datasets ", glob.glob(str(datasets_dir / "**/*.yaml"), recursive=True))
    print()
    for dataset_file in glob.glob(str(datasets_dir / "**/*.yaml"), recursive=True):
        if True:
            dataset = load_dataset(dataset_file)
            #print(dataset["metadata"]["category"], dataset_file)
            # Check if dataset matches task type
            if dataset["metadata"]["category"].lower() == task_type.lower():
                selected_datasets.append(dataset_file)
        else:
            print("error")
            continue
    #
    return selected_datasets


def run_single_evaluation(
    task_executor: Any,
    model_config: dict,
    evaluator_config: dict,
    task_name: str,
    test_cases: List[dict],
    output_file: str
) -> List[Dict[str, Any]]:
    """Run evaluation for a single test case or a list of test cases."""
                    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Create model
        try:
            model = create_model(
                model_name=model_config["model"]["name"],
                model_type=model_config["model"]["type"],
                **model_config.get("parameters", {})
            )
        except Exception as e:
            st.error(f"Failed to create model: {str(e)}")
            return []

        # Create evaluator
        try:
            eval_model = create_model(
                model_name=evaluator_config["model"]["name"],
                model_type=evaluator_config["model"]["type"],
                api_key_source=evaluator_config["model"]["api_key_source"],
                **evaluator_config.get("parameters", {})
            )
            evaluator = Evaluator(eval_model, evaluator_config["evaluator_prompt"])
        except Exception as e:
            st.error(f"Failed to create evaluator: {str(e)}")
            return []
        # Create task runner
        task_runner = task_executor(model)

        # Run task
        results = task_runner.run_task(test_cases)

        # Run evaluation
        evaluated_results = evaluator.evaluate_results(
            results,
            output_path=output_file
        )

        return evaluated_results

    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
        return []


def automatic_evaluation_page():
    """Page for running evaluations."""
    st.title("Automatic Evaluation")

    # Load configurations
    configs, msgs = load_config_files()
    for msg in msgs:
        st.error(msg)

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
    st.header("Automatic Evaluation")

    # Dataset-based evaluation
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        # Get available tasks from tasks directory
        available_tasks = get_available_tasks()
        
        if not available_tasks:
            st.error("No task implementations found in tasks directory")
        else:
            # Task selection (single task)
            selected_task = st.selectbox(
                "Select Task to Run",
                options=available_tasks,
                help="Choose which task to evaluate",
                key="dataset_task_select"
            )
            
            if not tasks_config.get(selected_task, None):
                st.error(f"Task **{selected_task}** not found in config file: **config/tasks.yaml**. Edit the config file!")
                help = """ 
                #### Example of task entry in `tasks.yaml`

                    name_of_task: 
                        import_lib: tasks.general_knowledge
                        executor: General_KnowlegeTask
                        dataset_path: datasets/instruction_following_ecommerce_tests.yaml
                        run_evaluation: true  
                        task_type: instruction_following

                Description:

                    `name_of_task`: (str) must be all lower cases and must not include string `task`
                    `import_lib`: (str) the module to import that has the implementatiton of the code to run the evaluation
                    `executor`: (str) name of the class to run the task
                    `dataset_path`: (str) path to dataset to use for this specific task
                    `run_evaluation`: (bool) whether to run model evaluation pipeline on the dataset
                    `task_type`: type of task
                """
                st.write(help)
                return
            
            # Show task documentation
            if selected_task:
                # Load TaskExecutor: SummarizationTask, etc...
                TaskExecutor = load_task_executor(module_path=tasks_config[selected_task]["import_lib"], 
                                                   class_name=tasks_config[selected_task]["executor"])
                doc = get_class_docstring(TaskExecutor)
                with st.expander("Task Description", expanded=True):
                    st.markdown(doc)
            
                # Dataset selection for selected task
                available_datasets = get_datasets_for_task(selected_task)
                
                if available_datasets:
                    selected_dataset = st.selectbox(
                        "Select Dataset",
                        options=available_datasets,
                        key=f"dataset_{selected_task}",
                        help=f"Select dataset to use for {selected_task} evaluation"
                    )
                    
                    if selected_dataset:
                        # Show dataset information
                        dataset = load_dataset(selected_dataset)
                        test_cases = dataset["test_cases"]
                        num_test_cases = len(test_cases)
                        if len(test_cases) > 0:
                            st.info(f"Number of test cases in dataset: {num_test_cases}")
                            with st.expander("Dataset Metadata", expanded=False):
                                st.json(dataset["metadata"])
                            
                            with st.expander("Test Cases", expanded=False):
                                st.json(test_cases)
                        else:
                            st.error(f"Error loading dataset or no test_cases in dataset: {selected_dataset}")
                    
                    st.session_state.selected_dataset = selected_dataset
                else:
                    st.warning(f"No datasets found for {selected_task}")
                    st.session_state.selected_dataset = None
    
    with settings_col2:
        output_file = st.text_input(
            "Save result to file:",
            "results/output.yaml",
            help="File path where the results will be saved",
            key="dataset_output_file"
        )
        if not output_file:
            st.error("Must enter a path name to save the results of the evaluation")
        if output_file.split(".")[-1].lower() != "yaml":
            st.error("the filename must be `yaml` extension")
            
    # Advanced settings in expander
    with st.expander("Advanced Settings"):
        show_progress_bar = st.checkbox(
            "Show Progress Bar",
            value=True,
            help="Display progress during evaluation"
        )
    
    # Validate settings before enabling run button
    can_run = selected_task and st.session_state.get('selected_dataset')
    
    # Run Evaluation Button
    if st.button("Run Evaluation", type="primary", disabled=not can_run):
        try:
            with st.spinner("Running evaluation..."):
                progress_bar = st.progress(0)
                st.write(f"Evaluating **{selected_task}** using dataset: **{st.session_state.selected_dataset}**")
                
                # Load dataset
                dataset = load_dataset(st.session_state.selected_dataset)
                # Run evaluation
                try:
                    start_timer = time.time()
                    results = run_single_evaluation(
                        task_executor=TaskExecutor,
                        model_config=model_config,
                        evaluator_config=evaluator_config,
                        task_name=selected_task,
                        test_cases=test_cases,
                        output_file=output_file
                    )
                    elapsed_time = time.time() - start_timer
                    elapsed_rate = elapsed_time / num_test_cases

                    # Show results
                    st.success(f"Evaluation completed! RunTime:{elapsed_time:.2f} sec ({elapsed_rate:.2f} secs per test case). Saved in: {output_file}")
                    st.subheader("Results")
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame([{
                        'Case ID': r.get('case_id', ''),
                        'Difficulty level': r.get('difficulty_level', ''),
                        'Score': r.get('score', 0),
                        'Feedback': r.get('feedback', '')
                    } for r in results])
                    
                    st.dataframe(results_df)
                    
                    # Display full results in an expandable section
                    with st.expander("Full Evaluation Results"):
                        st.json(results)
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                
                progress_bar.progress(1.0)
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")