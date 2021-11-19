import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

import yaml
from subprocess import Popen

from misc.utils import rm_and_mkdir

def write_yaml(data, path):
    with open(path, 'w') as f:
        stream = yaml.dump(data, default_flow_style=False)
        f.write(stream)

experiments_dir = os.path.join(os.getenv("INTRON_EXON_ROOT"), "experiments")
cnn_experiments_dir = os.path.join(experiments_dir, "cnn_v2_vary_dilation")

dilations = [1,2]

train_config_data = dict()
train_config_data['window_size'] = 1000
train_config_data['batch_size'] = 64
train_config_data['learning_rate'] = 1e-3
train_config_data['num_epochs'] = 80

experiment_num = 0
for dilation in dilations:
    curr_experiment_dir = os.path.join(cnn_experiments_dir, "experiment_" + str(experiment_num))
    rm_and_mkdir(curr_experiment_dir)

    cnn_config_data = dict()
    cnn_config_data['stride'] = 1
    cnn_config_data['dilation'] = dilation
    cnn_config_path = os.path.join(curr_experiment_dir, "cnn_v2_config.yml")
    write_yaml(cnn_config_data, cnn_config_path)

    train_config_path = os.path.join(curr_experiment_dir, "train_config.yml")
    write_yaml(train_config_data, train_config_path)

    os.chdir(os.path.join(os.getenv("INTRON_EXON_ROOT"), "train"))

    run_script = "python run.py --model CNNV2 --train_config " + train_config_path + " --model_config " + cnn_config_path + \
        " --output_dir " + curr_experiment_dir

    proc = Popen(run_script.split(' '))
    proc.wait()

    experiment_num += 1

