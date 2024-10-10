import yaml


def load_yaml(fname):
    # Load YAML file
    with open(fname, 'r') as file:
        data = yaml.safe_load(file)

    return data


def save_yaml(data, fname):
    # Save to a YAML file
    with open(fname, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)