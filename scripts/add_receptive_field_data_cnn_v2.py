import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

import yaml

def write_yaml(data, path):
    with open(path, 'w') as f:
        stream = yaml.dump(data, default_flow_style=False)
        f.write(stream)

def calculate_receptive_field_in_last_bottleneck_layer(stride, dilation):
    curr_receptive_field_size = 1
    for _ in range(4):
        curr_receptive_field_size = stride * curr_receptive_field_size + 9 * dilation
    return curr_receptive_field_size

experiments_dir = os.path.join(os.getenv("INTRON_EXON_ROOT"), "experiments")

def add_receptive_field_to_experiment(experiments_dir):
    for experiment_dir in os.listdir(experiments_dir):
        experiment_path = os.path.join(experiments_dir, experiment_dir)

        cnn_config_path = os.path.join(experiment_path, "cnn_v2_config.yml")
        with open(cnn_config_path, 'r') as f:
            cnn_config_data = yaml.safe_load(f)

        receptive_field = calculate_receptive_field_in_last_bottleneck_layer(cnn_config_data['stride'],
                                                                             cnn_config_data['dilation'])

        cnn_config_data['receptive_field'] = receptive_field
        write_yaml(cnn_config_data, cnn_config_path)

add_receptive_field_to_experiment(os.path.join(experiments_dir, "cnn_v2_vary_dilation"))
add_receptive_field_to_experiment(os.path.join(experiments_dir, "cnn_v2_vary_stride"))