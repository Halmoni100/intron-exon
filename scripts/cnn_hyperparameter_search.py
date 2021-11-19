import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

import itertools
import yaml
from subprocess import Popen

from misc.utils import rm_and_mkdir

def write_yaml(data, path):
    with open(path, 'w') as f:
        stream = yaml.dump(data, default_flow_style=False)
        f.write(stream)

experiments_dir = os.path.join(os.getenv("INTRON_EXON_ROOT"), "experiments")
cnn_experiments_dir = os.path.join(experiments_dir, "cnn")

channels = [(16, 8), (32, 16), (16, 16), (32, 32)]
kernel_sizes = [5, 7, 10]
num_bottleneck_convs = [0, 4, 16]
# bottleneck_dilation = [1,2]

train_config_data = dict()
train_config_data['window_size'] = 1000
train_config_data['batch_size'] = 64
train_config_data['learning_rate'] = 1e-3
train_config_data['num_epochs'] = 80

experiment_num = 0
for combination in itertools.product(channels, kernel_sizes, num_bottleneck_convs): # , bottleneck_dilation):
    curr_experiment_dir = os.path.join(cnn_experiments_dir, "experiment_" + str(experiment_num))
    rm_and_mkdir(curr_experiment_dir)

    cnn_config_data = dict()
    cnn_config_data['channels1'] = combination[0][0]
    cnn_config_data['channels2'] = combination[0][1]
    cnn_config_data['kernel_size'] = combination[1]
    cnn_config_data['num_bottleneck_convs'] = combination[2]
    cnn_config_path = os.path.join(curr_experiment_dir, "cnn_config.yml")
    write_yaml(cnn_config_data, cnn_config_path)

    train_config_path = os.path.join(curr_experiment_dir, "train_config.yml")
    write_yaml(train_config_data, train_config_path)

    os.chdir(os.path.join(os.getenv("INTRON_EXON_ROOT"), "train"))

    run_script = "python run.py --train_config " + train_config_path + " --model_config " + cnn_config_path + \
        " --output_dir " + curr_experiment_dir

    proc = Popen(run_script.split(' '))
    proc.wait()

    experiment_num += 1

