import streamlit as st
import textwrap
from typing import Dict, Any, Tuple

from lib.utils import load_config_files, save_config


def create_field_label(label: str, field_descriptions: dict) -> str:
    """Create field label."""
    return label


def handle_model_edit():
    st.session_state.model_edit_mode = True


def handle_evaluator_edit():
    st.session_state.evaluator_edit_mode = True


def handle_tasks_edit():
    st.session_state.tasks_edit_mode = True


def format_description(description: str) -> str:
    """Format description text to maintain line breaks."""
    if not description:
        return ""
    # Split description into lines and join with double newlines for markdown
    lines = [line.strip() for line in description.split('\n') if line.strip()]
    return '\n\n'.join(lines)


def edit_candidate_model_config(configs: Dict[str, Any], fname="candidate_model.yaml"):
    st.header("Candidate Model Configuration")
    st.caption(f"File: config/{fname}")

    model_config = configs.get(fname, {})
    field_descriptions = model_config.get("field_descriptions", {})

    with st.expander("About Candidate Model Configuration", expanded=False):
        description = format_description(model_config.get("description", ""))
        st.markdown(description)

    col1, col2 = st.columns([3, 1])

    with col1:
        model = model_config.get("model", {})
        parameters = model_config.get("parameters", {})

        # Dynamically create input fields for model and parameters
        for section_name, section_config in [("model", model), ("parameters", parameters)]:
            for field_name, field_value in section_config.items():
                if isinstance(field_value, (int, float, str)):
                    if isinstance(field_value, int):
                        input_func = st.number_input
                        value = int(field_value)
                    elif isinstance(field_value, float):
                        input_func = st.number_input
                        value = float(field_value)
                    else:
                        input_func = st.text_input
                        value = str(field_value)

                    section_config[field_name] = input_func(
                        create_field_label(field_name, field_descriptions),
                        value=value,
                        disabled=not st.session_state.model_edit_mode,
                        key=f"{section_name}_{field_name}_input",
                        help=field_descriptions.get(field_name, "")
                    )

    with col2:
        if not st.session_state.model_edit_mode:
            st.button(
                "Edit",
                key="model_edit_btn",
                on_click=handle_model_edit,
                use_container_width=True
            )
        else:
            if st.button("Save", key="model_save_btn", use_container_width=True):
                new_config = model_config.copy()
                new_config["model"] = model
                new_config["parameters"] = parameters
                msg_type, msg = save_config(fname, new_config)
                st.success(msg) if msg_type == "success" else st.error(msg)
                st.session_state.model_edit_mode = False
                st.experimental_rerun()


def edit_evaluator_config(configs: Dict[str, Any], fname="evaluator.yaml"):
    st.header("Evaluator Model Configuration")
    st.caption(f"File: config/{fname}")

    model_config = configs.get(fname, {})
    field_descriptions = model_config.get("field_descriptions", {})

    with st.expander("About Evaluator Model Configuration", expanded=False):
        description = format_description(model_config.get("description", ""))
        st.markdown(description)

    col1, col2 = st.columns([3, 1])

    with col1:
        model = model_config.get("model", {})
        parameters = model_config.get("parameters", {})
        evaluator_prompt = model_config.get("evaluator_prompt", "")

        # Dynamically create input fields for model and parameters
        for section_name, section_config in [("model", model), ("parameters", parameters)]:
            for field_name, field_value in section_config.items():
                if isinstance(field_value, (int, float, str)):
                    if isinstance(field_value, int):
                        input_func = st.number_input
                        value = int(field_value)
                    elif isinstance(field_value, float):
                        input_func = st.number_input
                        value = float(field_value)
                    else:
                        input_func = st.text_input
                        value = str(field_value)

                    section_config[field_name] = input_func(
                        create_field_label(field_name, field_descriptions),
                        value=value,
                        disabled=not st.session_state.model_edit_mode,
                        key=f"{section_name}_{field_name}_input",
                        help=field_descriptions.get(field_name, "")
                    )
                    
                # Handle the evaluator_prompt separately
        evaluator_prompt_input = st.text_area(
            create_field_label("evaluator_prompt", field_descriptions),
            value=evaluator_prompt,
            disabled=not st.session_state.model_edit_mode,
            key="evaluator_prompt_input",
            height=300,
            help=field_descriptions.get("evaluator_prompt", "")
        )
        model_config["evaluator_prompt"] = textwrap.dedent(evaluator_prompt_input)

    with col2:
        if not st.session_state.model_edit_mode:
            st.button(
                "Edit",
                key="model_edit_btn",
                on_click=handle_model_edit,
                use_container_width=True
            )
        else:
            if st.button("Save", key="model_save_btn", use_container_width=True):
                new_config = model_config.copy()
                new_config["model"] = model
                new_config["parameters"] = parameters
                msg_type, msg = save_config(fname, new_config)
                st.success(msg) if msg_type == "success" else st.error(msg)
                st.session_state.model_edit_mode = False
                st.experimental_rerun()

        
def edit_tasks_config(configs: Dict[str, Any], fname="tasks.yaml"):
    st.header("Tasks Configuration")
    st.caption(f"File: config/{fname}")

    tasks_config = configs.get(fname, {})
    field_descriptions = tasks_config.get("field_descriptions", {})

    with st.expander("About Tasks Configuration", expanded=False):
        description = format_description(tasks_config.get("description", ""))
        st.markdown(description)

    col1, col2 = st.columns([3, 1])

    with col1:
        tasks = tasks_config.get("tasks", {})
        # Dynamically create input fields for model and parameters
        for task_name, task_config in tasks.items():
            for field_name, field_value in task_config.items():
                if isinstance(field_value, (int, float, str, bool)):
                    if isinstance(field_value, bool):
                        input_func = st.checkbox
                        value = bool(field_value)
                    elif isinstance(field_value, int):
                        input_func = st.number_input
                        value = int(field_value)
                    elif isinstance(field_value, float):
                        input_func = st.number_input
                        value = float(field_value)
                    else:
                        input_func = st.text_input
                        value = str(field_value)

                    task_config[field_name] = input_func(
                        create_field_label(field_name, field_descriptions),
                        value=value,
                        disabled=not st.session_state.model_edit_mode,
                        key=f"{task_name}_{field_name}_input",
                        help=field_descriptions.get(field_name, "")
                    )

    with col2:
        if not st.session_state.model_edit_mode:
            st.button(
                "Edit",
                key="tasks_edit_btn",
                on_click=handle_model_edit,
                use_container_width=True
            )
        else:
            if st.button("Save", key="tasks_save_btn", use_container_width=True):
                new_config = tasks_config.copy()
                new_config["tasks"] = tasks
                msg_type, msg = save_config(fname, new_config)
                st.success(msg) if msg_type == "success" else st.error(msg)
                st.session_state.model_edit_mode = False
                st.experimental_rerun()


def config_editor_page():
    """Page for editing configuration files."""
    st.title("Configuration Editor")
    
    # Initialize session states
    if 'model_edit_mode' not in st.session_state:
        st.session_state.model_edit_mode = False
    if 'evaluator_edit_mode' not in st.session_state:
        st.session_state.evaluator_edit_mode = False
    if 'tasks_edit_mode' not in st.session_state:
        st.session_state.tasks_edit_mode = False
    
    # Load configurations
    configs, msgs = load_config_files()
    for msg in msgs:
        st.error(msg)
    
    # Create section selection
    current_tab = st.selectbox(
        "Select Section",
        ["Candidate Model", "Evaluator", "Tasks"],
        label_visibility="collapsed"
    )

    if current_tab == "Candidate Model":
        edit_candidate_model_config(configs)
    elif current_tab == "Evaluator":
        edit_evaluator_config(configs)
    else:  # Tasks tab
        edit_tasks_config(configs)