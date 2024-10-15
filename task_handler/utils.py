import os
import yaml
from collections import OrderedDict


def load_yaml(fname):
    """
    Load and parse a YAML file.

    This function reads a YAML file from the specified file path and 
    returns its contents as a Python data structure, typically a dictionary 
    or list, depending on the structure of the YAML file.

    Args:
        fname (str): The file path to the YAML file to be loaded.

    Returns:
        Any: The contents of the YAML file as a Python data structure.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.

    Notes:
        - The function uses `yaml.safe_load` to parse the YAML file, which 
          is a safe way to load YAML content without executing arbitrary 
          code.
    """
    if os.path.exists(fname):
        raise FileNotFoundError(f"{fname}")
    
    with open(fname, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data, fname):
    """
    Save a Python data structure to a YAML file.

    This function writes the provided data to a YAML file at the specified 
    file path. The data is serialized into YAML format, making it suitable 
    for configuration files or data exchange.

    Args:
        data (Any): The Python data structure (e.g., dictionary or list) to 
            be saved to the YAML file.
        fname (str): The file path where the YAML file will be saved.

    Returns:
        None: This function does not return a value. It writes the data to 
        the specified file.

    Notes:
        - The function uses `yaml.dump` to serialize the data into YAML 
          format.
        - The `default_flow_style=False` parameter ensures that the YAML 
          output is in block style, which is more human-readable.
    """
    with open(fname, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


def custom_sort(dictionary, ls_keys):
    """
    Create a new dictionary sorted by a specified list of keys.

    This function takes a dictionary and a list of keys, and returns a new 
    dictionary containing only the key-value pairs from the original 
    dictionary that match the keys in the list, in the order specified by 
    the list.

    Args:
        dictionary (Dict[Any, Any]): The original dictionary to be sorted.
        ls_keys (List[Any]): A list of keys specifying the order and subset 
            of the dictionary to include in the new dictionary.

    Returns:
        Dict[Any, Any]: A new dictionary containing key-value pairs from the 
        original dictionary, sorted according to the order of `ls_keys`.

    Notes:
        - If a key in `ls_keys` does not exist in the original dictionary, 
          it is ignored.
        - This function is useful for reordering or filtering dictionaries 
          based on a predefined key order.
    """
    return {k: dictionary[k] for k in ls_keys if k in dictionary}


class BlankLineDumper(yaml.Dumper):
    """
    A custom YAML Dumper that inserts blank lines between top-level keys.

    This class extends the default `yaml.Dumper` to add an extra line break 
    between top-level keys in the YAML output. This can improve readability 
    by visually separating sections of the YAML file.

    Methods:
        write_line_break(data=None): Overrides the default line break 
        behavior to insert an additional blank line between top-level keys.

    Usage:
        Use `BlankLineDumper` as the `Dumper` argument in `yaml.dump` to 
        apply this formatting to the YAML output.

    Example:
        yaml.dump(data, Dumper=BlankLineDumper)
    """
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:  # Adds a blank line between top-level keys
            super().write_line_break()


def save_yaml_custom_sort(data, fname):
    """
    Save a Python data structure to a YAML file with custom formatting.

    This function writes the provided data to a YAML file at the specified 
    file path, using a custom YAML dumper that inserts blank lines between 
    top-level keys. The data is serialized into YAML format in a human-readable 
    block style, and the keys are not sorted, preserving the original order.

    Args:
        data (Any): The Python data structure (e.g., dictionary or list) to 
            be saved to the YAML file.
        fname (str): The file path where the YAML file will be saved.

    Returns:
        None: This function does not return a value. It writes the data to 
        the specified file.

    Notes:
        - The function uses `BlankLineDumper` to format the YAML output with 
          blank lines between top-level keys for improved readability.
        - The `default_flow_style=False` parameter ensures that the YAML 
          output is in block style.
        - The `sort_keys=False` parameter preserves the order of keys as they 
          appear in the input data.
    """
    with open(fname, 'w') as file:
        yaml.dump(data, file, Dumper=BlankLineDumper, default_flow_style=False, sort_keys=False)


def validate_test_dataset(dataset, required_keys):
    """
    Validate that each record in a dataset contains all required keys.

    This function checks each record in the provided dataset to ensure that 
    it contains all the keys specified in the `required_keys` list. If any 
    record is missing a required key, a message is printed, and an assertion 
    error is raised if any records are invalid.

    Args:
        dataset (List[Dict[str, Any]]): A list of records, where each record 
            is a dictionary representing a test case or data entry.
        required_keys (List[str]): A list of keys that each record in the 
            dataset must contain.

    Returns:
        None: This function does not return a value. It raises an assertion 
        error if any records are missing required keys.

    Raises:
        AssertionError: If one or more records in the dataset are missing 
        required keys, an assertion error is raised with a message indicating 
        the issue.

    Notes:
        - The function prints a message for each missing key in a record, 
          helping to identify which keys are missing from which records.
        - The assertion ensures that the dataset is valid before proceeding 
          with further processing or analysis.
    """
    n_failed = 0
    for record in dataset:
        record_ks = set(list(record.keys()))
        for k in required_keys:
            if k not in record_ks:
                print("[Not valid test, Missing key=`{k}` in record")
                n_failed += 1

    assert n_failed == 0, f"[Error] Found missing keys in dataset"

    return


def create_directory(output_dir: str) -> None:
    """
    Create an output directory if it does not already exist.

    This function checks if the root path of the specified output directory 
    exists. If the root path does not exist, it raises a FileNotFoundError. 
    If the output directory itself does not exist, the function attempts to 
    create it. If the directory is successfully created, a confirmation 
    message is printed. If the directory already exists, a message is printed 
    indicating this. If there is a permission issue while creating the 
    directory, a PermissionError is raised.

    Args:
        output_dir (str): The path of the directory to be created.

    Returns:
        None: This function does not return a value. It either creates the 
        directory or raises an error if it cannot be created.

    Raises:
        FileNotFoundError: If the root path of the specified output directory 
        does not exist.
        PermissionError: If there is a permission issue preventing the 
        creation of the directory.

    Notes:
        - The function uses `os.makedirs` to create the directory, which 
          allows for the creation of intermediate directories if necessary.
        - It is important to ensure that the root path exists before calling 
          this function to avoid unexpected errors.
    """
    root_path = os.path.dirname(output_dir)
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except PermissionError:
            raise PermissionError(f"Permission denied: Unable to create directory {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")