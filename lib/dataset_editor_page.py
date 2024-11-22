import streamlit as st
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import date

from lib.utils import save_dataset


def upload_dataset():
    uploaded_file = st.file_uploader(
        "Upload YAML Dataset",
        type=['yaml', 'yml'],
        key="dataset_file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            parsed_data = yaml.safe_load(content)
            st.session_state.dataset_metadata = parsed_data.get("metadata", {})
            st.session_state.dataset = parsed_data.get("dataset", [])
            st.success(f"File uploaded successfully! Loaded {len(st.session_state.dataset)} cases.")

                        # Display the content of the dataset
            st.subheader("Dataset Metadata")
            with st.expander(f"Metadata"):
                st.json(st.session_state.dataset_metadata)
            
            # Display the content of the dataset
            st.subheader("Dataset Examples")
            for i, case in enumerate(st.session_state.dataset):
                with st.expander(f"Case {case.get('case_id', '')}"):
                    st.json(case)
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")


def create_new_dataset():
    # Dataset Metadata Section
    metadata_expanded = len(st.session_state.new_dataset) == 0
    
    with st.expander("Dataset Metadata", expanded=metadata_expanded):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.new_dataset_metadata["name"] = st.text_input(
                "Dataset Name",
                value=st.session_state.new_dataset_metadata.get("name", "")
            )
            st.session_state.new_dataset_metadata["category"] = st.text_input(
                "Category",
                value=st.session_state.new_dataset_metadata.get("category", "")
            )
            st.session_state.new_dataset_metadata["version"] = st.text_input(
                "Version",
                value=st.session_state.new_dataset_metadata.get("version", "1.0")
            )
        
        with col2:
            st.session_state.new_dataset_metadata["author"] = st.text_input(
                "Author",
                value=st.session_state.new_dataset_metadata.get("author", "")
            )
            st.session_state.new_dataset_metadata["created_date"] = st.date_input(
                "Created Date",
                value=date.today()
            ).strftime("%Y-%m-%d")
        
        st.session_state.new_dataset_metadata["description"] = st.text_area(
            "Description",
            value=st.session_state.new_dataset_metadata.get("description", "")
        )
        
        tags_str = st.text_input(
            "Tags (comma-separated)",
            value=",".join(st.session_state.new_dataset_metadata.get("tags", []))
        )
        st.session_state.new_dataset_metadata["tags"] = [
            tag.strip() for tag in tags_str.split(",") if tag.strip()
        ]
        
        metrics_str = st.text_input(
            "Metrics (comma-separated)",
            value=",".join(st.session_state.new_dataset_metadata.get("metrics", []))
        )
        st.session_state.new_dataset_metadata["metrics"] = [
            metric.strip() for metric in metrics_str.split(",") if metric.strip()
        ]

    # Add Test Case Section
    with st.expander("Add New Test Case", expanded=True):
        with st.form(key="new_case_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.new_dataset_last_case["category"] = st.text_input(
                    "Category",
                    value=st.session_state.new_dataset_metadata.get("category", ""),
                    disabled=True
                )
                st.session_state.new_dataset_last_case["sub_category"] = st.text_input(
                    "Sub-category",
                    value=st.session_state.new_dataset_last_case.get("sub_category", "")
                )

            with col2:
                st.session_state.new_dataset_last_case["difficulty_level"] = st.selectbox(
                    "Difficulty Level",
                    options=["easy", "medium", "hard"]
                )
            
            st.session_state.new_dataset_last_case["system_prompt"] = st.text_area(
                "System Prompt",
                value=st.session_state.new_dataset_last_case.get("system_prompt", 
                                                                 "You are an expert at creating concise summaries..."),
                height=100
            )
            
            st.session_state.new_dataset_last_case["instruction"] = st.text_area(
                "Instruction",
                value=st.session_state.new_dataset_last_case.get("instruction", ""),
                height=100
            )

            st.session_state.new_dataset_last_case["expected_response"] = st.text_area(
                "Expected Response",
                value=st.session_state.new_dataset_last_case.get("expected_response", ""), 
                height=100
            )
            
            st.session_state.new_dataset_last_case["potential_challenges"] = st.text_area(
                "Potential Challenges",
                value=st.session_state.new_dataset_last_case.get("potential_challenges", ""),
                height=100
            )
            
            submitted = st.form_submit_button("Add Test Case")
            
            if submitted:
                case_id = len(st.session_state.new_dataset) + 1
                new_case = {
                    "case_id": case_id,
                    "category": st.session_state.new_dataset_metadata.get("category", ""),
                    "sub_category": st.session_state.new_dataset_metadata.get("sub_category", ""),
                    "difficulty_level": st.session_state.new_dataset_last_case["difficulty_level"],
                    "system_prompt": st.session_state.new_dataset_last_case["system_prompt"],
                    "instruction": st.session_state.new_dataset_last_case["instruction"],
                    "expected_response": st.session_state.new_dataset_last_case["expected_response"],
                    "potential_challenges": st.session_state.new_dataset_last_case["potential_challenges"]
                }
                st.session_state.new_dataset.append(new_case)
                st.session_state.new_dataset_last_case = new_case
                st.success(f"Test case [{case_id}] added successfully!")

    # Dataset Overview Section
    if st.session_state.new_dataset:
        st.header(f"Current Dataset (size={len(st.session_state.new_dataset)})")
        for i, case in enumerate(st.session_state.new_dataset):
            with st.expander(f"Case {case.get('case_id', '')}"):
                st.json(case)
        
        # Save Dataset Section
        st.header("Save Dataset")
        
        save_path = st.text_input(
            "Save Path",
            "datasets/new_dataset.yaml"
        )
        
        if st.button("Save Dataset"):
            try:
                dataset = {
                    "metadata": st.session_state.new_dataset_metadata,
                    "dataset": st.session_state.new_dataset
                }
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                msg_type, msg = save_dataset(save_path, dataset)
                st.success(msg) if msg_type == "success" else st.error(msg)
            except Exception as e:
                st.error(f"Error saving dataset: {e}")


def dataset_builder_page():
    """Page for building evaluation datasets."""
    st.title("Dataset Builder")
    
    # Initialize session state
    if 'dataset_metadata' not in st.session_state:
        st.session_state.dataset_metadata = {}

    if 'new_dataset' not in st.session_state:
        st.session_state.new_dataset = []
    
    if 'new_dataset_last_case' not in st.session_state:
        st.session_state.new_dataset_last_case = {}
    
    if 'new_dataset_metadata' not in st.session_state:
        st.session_state.new_dataset_metadata = {}

    if 'dataset' not in st.session_state:
        st.session_state.dataset = []
    
    if 'current_case_index' not in st.session_state:
        st.session_state.current_case_index = 0
        
    if 'operation_mode' not in st.session_state:
        st.session_state.operation_mode = None

    # Dataset Operations Section
    st.header("Dataset Operations")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Upload Dataset", use_container_width=True):
            st.session_state.operation_mode = "upload"
            st.session_state.dataset = []
            st.experimental_rerun()
            
    with col2:
        if st.button("Create New", use_container_width=True):
            st.session_state.operation_mode = "create"
            st.session_state.new_dataset = []
            st.session_state.new_dataset_last_case = {}
            st.session_state.new_dataset_metadata = {}
            st.experimental_rerun()

    if st.session_state.operation_mode == "upload":
        upload_dataset()
    elif st.session_state.operation_mode == "create":
        create_new_dataset()