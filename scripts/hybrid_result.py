import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

from subprocess import Popen
import yaml

from misc.utils import rm_and_mkdir

def write_yaml(data, path):
    with open(path, 'w') as f:
        stream = yaml.dump(data, default_flow_style=False)
        f.write(stream)

experiments_dir = os.path.join(os.getenv("INTRON_EXON_ROOT"), "experiments")
hybrid_experiment_dir = os.path.join(experiments_dir, "hybrid")

train_config_data = dict()
train_config_data['window_size'] = 1000
train_config_data['batch_size'] = 64
train_config_data['learning_rate'] = 1e-3
train_config_data['num_epochs'] = 50

rm_and_mkdir(hybrid_experiment_dir)

hybrid_config_data = dict()
hybrid_config_path = os.path.join(hybrid_experiment_dir, "hybrid_config.yml")
write_yaml(hybrid_config_data, hybrid_config_path)

train_config_path = os.path.join(hybrid_experiment_dir, "train_config.yml")
write_yaml(train_config_data, train_config_path)

os.chdir(os.path.join(os.getenv("INTRON_EXON_ROOT"), "train"))

run_script = "python run.py --model HYBRID --train_config " + train_config_path + " --model_config " + hybrid_config_path + \
             " --output_dir " + hybrid_experiment_dir

proc = Popen(run_script.split(' '))
proc.wait()

