import yaml
from pathlib import Path
import argparse
from typing import Dict, Any, List


def validate_yaml_file(file_path: str) -> None:
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # Check that the required keys exist
    if 'metadata' not in data or 'test_cases' not in data:
        print("Error: 'metadata' and 'test_cases' keys are required.")
        return

    # Validate metadata
    metadata = data['metadata']
    if 'description' not in metadata or 'category' not in metadata:
        print("Error: 'description' and 'category' keys are required in the metadata.")
        return

    if 'task' in metadata['category'].lower():
        val = metadata['category']
        print("Error: 'category' value must not contain the substring 'task', got {val}")
        return

    if not isinstance(metadata['category'], str) or not metadata['category'].islower():
        val = metadata['category']
        print(f"Error: 'category' value must be a lowercase string, got {val}")

        return

    # Validate test cases
    test_cases = data['test_cases']
    if not test_cases:
        print("Error: 'test_cases' list must not be empty.")
        return

    for case in test_cases:
        # Check required keys
        required_keys = ['case_id', 'category', 'system_prompt', 'expected_response', 'difficulty_level']
        for key in required_keys:
            if key not in case:
                print(f"Error: '{key}' is a required key in the test case.")
                return

        # Validate case_id
        if not isinstance(case['case_id'], int):
            print(f"Error: 'case_id' must be an integer, got {type(case['case_id'])}.")
            return

        # Validate category
        if case['category'] != metadata['category']:
            val = metadata['category']
            print(f"Error: 'category' value must match the metadata 'category' value, got {val}")
            return

        # Validate system_prompt and expected_response
        for field in ['system_prompt', 'expected_response']:
            if not isinstance(case[field], str):
                print(f"Error: '{field}' must be a string, got {type(case[field])}.")
                return

        # Validate challenges
        if 'challenges' in case and not isinstance(case['challenges'], str):
            val = case['challenges']
            print("Error: 'challenges' must be a string or empty string, got {val}")
            return

        # Validate difficulty_level
        if case['difficulty_level'] not in ['easy', 'medium', 'hard']:
            val = case['difficulty_level']
            print("Error: 'difficulty_level' must be 'easy', 'medium', or 'hard', got {val}")
            return

    print(f"\nYML dataset file is valid: {file_path}: ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a YAML file')
    parser.add_argument('--file_path', type=str, help='Path to the YAML file')
    args = parser.parse_args()

    validate_yaml_file(args.file_path)