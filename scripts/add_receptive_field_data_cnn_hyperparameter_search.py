import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

import yaml

def write_yaml(data, path):
    with open(path, 'w') as f:
        stream = yaml.dump(data, default_flow_style=False)
        f.write(stream)

def calculate_receptive_field_in_last_bottleneck_layer(kernel_size, num_bottleneck_convs):
    curr_receptive_field_size = 1
    for _ in range(num_bottleneck_convs):
        curr_receptive_field_size += kernel_size - 1
    for _ in range(2):
        curr_receptive_field_size = 2*curr_receptive_field_size + kernel_size - 1
    return curr_receptive_field_size

experiments_dir = os.path.join(os.getenv("INTRON_EXON_ROOT"), "experiments")
cnn_experiments_dir = os.path.join(experiments_dir, "cnn")

for experiment_dir in os.listdir(cnn_experiments_dir):
    experiment_path = os.path.join(cnn_experiments_dir, experiment_dir)

    cnn_config_path = os.path.join(experiment_path, "cnn_config.yml")
    with open(cnn_config_path, 'r') as f:
        cnn_config_data = yaml.safe_load(f)

    receptive_field = calculate_receptive_field_in_last_bottleneck_layer(cnn_config_data['kernel_size'],
                                                                         cnn_config_data['num_bottleneck_convs'])

    cnn_config_data['receptive_field'] = receptive_field
    write_yaml(cnn_config_data, cnn_config_path)

