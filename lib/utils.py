import os
import re
from glob import glob
import ruamel.yaml
from pathlib import Path
from typing import Dict, Any, List


def load_yaml(fname):
    yaml = ruamel.yaml.YAML()
    with open(fname) as f:
        data = yaml.load(f)
    
    return data


def load_config_files(config_dir="config") -> Dict[str, Any]:
    """Load all configuration files from the config directory."""

    config_dir = Path(config_dir)
    configs = {}
    error_msgs = []
    for config_file in ["candidate_model.yaml", "evaluator.yaml", "tasks.yaml"]:
        try:
            fname = config_dir / config_file
            configs[config_file] = load_yaml(fname)
        except Exception as e:
            error_msgs.append(f"Error loading {config_file}: {e}")
            configs[config_file] = {}
    
    return configs, error_msgs


def save_config(fname: str, config: Dict[str, Any]) -> None:
    """Save the configuration to the specified file.
    Use ruamel.yaml so that multiline formatting is preserved
    """
    try:
        config_dir = Path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / fname

        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.width = 120

        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        msg = f"Successfully saved {fname}"
        msg_type = "success"
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        msg = f"Error saving {fname}: {e}"
        msg_type = "error"

    return msg_type, msg


def save_dataset(path_to_fname: str, dataset: Dict):
    """Save the dataset as yaml file
    Use ruamel.yaml so that multilines formatting is preserved
    """
    pattern = r"case_id:\s*\d+"  # case_id

    try:
        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.default_flow_style = False
        yaml.width = 120
        with open(path_to_fname, 'w') as f:
            yaml.dump(dataset, f)
        
        # Add a blank line manually between 'metadata' and 'dataset'
        with open(path_to_fname, 'r') as yaml_file:
            lines = yaml_file.readlines()

        with open(path_to_fname, 'w') as yaml_file:
            for i, line in enumerate(lines):
                # Insert blank line after 'metadata' block
                if line.strip() == 'dataset:':
                    yaml_file.write("\n\n")
                if re.search(pattern, line):
                    # Insert blank line between case_ids
                    yaml_file.write("\n")
                yaml_file.write(line)

        msg = f"Successfully saved {path_to_fname}"
        msg_type = "success"
    except Exception as e:
        msg = f"Error saving {path_to_fname}: {e}"
        msg_type = "error"

    return msg_type, msg


def get_available_tasks() -> List[str]:
    """Scan tasks folder and return available task names."""
    tasks_dir = Path("tasks")
    if not tasks_dir.exists():
        return []
    
    # Get all Python files in the tasks directory, excluding __init__.py and common files
    task_files = [
        f.stem for f in tasks_dir.glob("*.py") 
        if f.name != "__init__.py" and not f.name.startswith("_")
    ]
    
    return task_files


def load_dataset(fname):
    # Load dataset: 
    data = load_yaml(fname)
    return {"test_cases": data.get("test_cases", []), 
            "metadata": data.get("metadata", {})}
