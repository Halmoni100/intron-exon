import os
import itertools
import pathlib
import yaml

from utils import rm_and_mkdir, run_bash_script

def write_yaml(data, path):
    with open(path, 'w') as f:
        stream = yaml.dump(data, default_flow_style=False)
        f.write(stream)

experiments_dir = os.path.join(os.environ['INTRON_EXON_ROOT'], "experiments")
temp_dir = os.path.join(os.environ['INTRON_EXON_ROOT'], "temp")
rnn_experiments_dir = os.path.join(experiments_dir, "rnn")

num_stacks = [1,3]

train_config_data = dict()
train_config_data['window_size'] = 1000
train_config_data['batch_size'] = 64
train_config_data['learning_rate'] = 1e-3
train_config_data['num_epochs'] = 160

experiment_num = 0
for n in num_stacks: # , bottleneck_dilation):
    curr_experiment_dir = os.path.join(rnn_experiments_dir, "experiment_" + str(experiment_num))
    rm_and_mkdir(curr_experiment_dir)

    rnn_config_data = dict()
    rnn_config_data['num_stacks'] = n
    rnn_config_path = os.path.join(curr_experiment_dir, "rnn_config.yml")
    write_yaml(rnn_config_data, rnn_config_path)

    train_config_path = os.path.join(curr_experiment_dir, "train_config.yml")
    write_yaml(train_config_data, train_config_path)

    run_script = "cd " + os.environ['INTRON_EXON_ROOT'] + "\n"
    run_script += "python train/run.py --model RNN --train_config " + train_config_path + " --model_config " + rnn_config_path + \
        " --output_dir " + curr_experiment_dir
    run_bash_script(run_script, temp_dir)

    experiment_num += 1

