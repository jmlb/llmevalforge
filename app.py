import streamlit as st
import os
import yaml
from typing import Dict, Any
import importlib

# Import functions from _main.py
from dotenv import load_dotenv
from _main import create_model, load_task, get_api_key, load_yaml, save_yaml
from model_handler.evaluators import run_evaluation
# Import the summarization task
from task_handler.summarization import run_summarization_task


load_dotenv()

st.set_page_config(layout="wide")
st.title("Model Evaluation App")

def run_model_on_example(model, example: Dict[str, Any], task_config: Dict[str, Any]) -> str:
    task_func = run_summarization_task  # We're using the summarization task directly
    result = task_func(model, [example], **task_config)
    return result[0]['response_candidate_model']

def run_evaluator_on_example(evaluator, example: Dict[str, Any], model_output: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
    example_with_output = example.copy()
    example_with_output['response_candidate_model'] = model_output
    result = evaluator([example_with_output], **task_config)
    return result[0]

# Sidebar for configuration
st.sidebar.header("Configuration")

# 1 & 7. Load local YAML file
config_file = st.sidebar.file_uploader("Upload config YAML file", type="yaml")

if config_file is not None:
    config = yaml.safe_load(config_file)
    st.sidebar.success("Config file loaded successfully!")

    # Display the name of the student model
    student_model_name = config['candidate_model']['params']['model']
    st.sidebar.write(f"Student Model: **{student_model_name}**")

    # 2. Select example task file
    task_files = ["Random"] + list(config['evaluation_tasks'].keys())
    task_file = st.sidebar.selectbox("Select task file", task_files)
    
    if task_file:
        if task_file == "Random":
            # For Random option, create an empty example
            dataset = [{
                "case_id": 1,
                "system_prompt": '',
                "instruction": '',
                "expected_response": 'N/A for random prompts'
            }]
            task_config = {}
        else:
            task_config = config['evaluation_tasks'][task_file]
            dataset = load_yaml(task_config['dataset_file'])
            
        # Display examples on the sidebar
        st.sidebar.header("Examples")
        example_options = [f"Case {example['case_id']}" for example in dataset]
        selected_example = st.sidebar.radio("Select an example", example_options)
            
        # Main area
        st.header("Model Evaluation")
        
        # Get the selected example data
        selected_example_data = next(example for example in dataset if f"Case {example['case_id']}" == selected_example)
        
        # Create two columns: one for example details and one for model evaluation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Example Details")
            # Editable text area for system prompt
            system_prompt = st.text_area("System Prompt", value=selected_example_data['system_prompt'], height=150)
            
            # Editable text area for instruction
            instruction = st.text_area("Instruction", value=selected_example_data['instruction'], height=150)
            
            # Update the selected_example_data with potentially edited values
            selected_example_data['system_prompt'] = system_prompt
            selected_example_data['instruction'] = instruction
        
        with col2:
            st.subheader("Model Evaluation")
            
            # 3 & 4. Button to run test model on selected example
            if st.button("Run Test Model"):
                with st.spinner("Running model..."):
                    # Initialize model
                    model = create_model(config['candidate_model'])
                    
                    # Run model on selected example
                    model_output = run_model_on_example(model, selected_example_data, task_config)
                    # 5. Display results
                    st.write("Expected Output:")
                    st.text_area("", value=selected_example_data['expected_response'], height=150, disabled=True)
                    
                    st.write("Model Output:")
                    st.text_area("", value=selected_example_data["response_candidate_model"], height=150, disabled=True)


                    # Save model output for later evaluation
                    st.session_state.model_output = model_output
                    st.session_state.selected_example_data = selected_example_data
                    
            
            # 6. Button to run evaluator
            if st.button("Run Evaluator"):
                if 'model_output' in st.session_state and 'selected_example_data' in st.session_state:
                    with st.spinner("Running evaluator..."):
                        example_data = st.session_state.selected_example_data
                        evaluation_result = run_evaluation([example_data], **config["evaluator"])
                        evaluation_result = evaluation_result[0]
                        st.subheader("Evaluation Results")
                        st.write(f"Score: {evaluation_result['score']}")
                        st.write("Score Feedback:")
                        st.text_area("", value=evaluation_result['scorer_feedback'], height=150, disabled=True)
                else:
                    st.warning("Please run the test model first.")

else:
    st.warning("Please upload a config file to get started.")

# 7. Button to browse current folder and select YAML
if st.sidebar.button("Browse Local YAML Files"):
    yaml_files = [f for f in os.listdir('.') if f.endswith('.yaml') or f.endswith('.yml')]
    selected_yaml = st.sidebar.selectbox("Select a YAML file", yaml_files)
    if selected_yaml:
        with open(selected_yaml, 'r') as file:
            st.sidebar.text_area("YAML Content", file.read(), height=300)