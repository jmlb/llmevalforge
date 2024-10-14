import yaml
from collections import OrderedDict


def load_yaml(fname):
    # Load YAML file    
    # Load the YAML file
    with open(fname, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data, fname):
    # Save to a YAML file
    with open(fname, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


def custom_sort(dictionary, ls_keys):
    return {k: dictionary[k] for k in ls_keys if k in dictionary}


# Custom YAML Dumper that adds a blank line between entries
class BlankLineDumper(yaml.Dumper):
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:  # Adds a blank line between top-level keys
            super().write_line_break()


def save_yaml_custom_sort(data, fname):
    with open(fname, 'w') as file:
        yaml.dump(data, file, Dumper=BlankLineDumper, default_flow_style=False, sort_keys=False)


def validate_test_dataset(dataset, required_keys):
    n_failed = 0
    for record in dataset:
        record_ks = set(list(record.keys()))
        for k in required_keys:
            if k not in record_ks:
                print("[Not valid test, Missing key=`{k}` in record")
                n_failed += 1

    assert n_failed == 0, f"[Error] Found missing keys in dataset"

    return