import os

import yaml
import pandas as pd

columns = [
    # dir
    'directory',
    # train config
    'batch_size',
    'learning_rate',
    'num_epochs',
    'window_size',
    # cnn config
    'channels1',
    'channels2',
    'kernel_size',
    'num_bottleneck_convs',
    # results
    'end_accuracy',
    'max_accuracy'
]

def get_dataframe():
    df = pd.DataFrame(columns=columns)

    cnn_experiments_dir = os.path.join(os.getenv("INTRON_EXON_ROOT"), "experiments", "cnn")
    for experiment_dir in os.listdir(cnn_experiments_dir):
        experiment_path = os.path.join(cnn_experiments_dir, experiment_dir)
        entries = list()
        entries.append(experiment_dir)

        train_config_path = os.path.join(experiment_path, 'train_config.yml')
        if not os.path.exists(train_config_path):
            continue
        with open(train_config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        entries.append(train_config['batch_size'])
        entries.append(train_config['learning_rate'])
        entries.append(train_config['num_epochs'])
        entries.append(train_config['window_size'])

        cnn_config_path = os.path.join(experiment_path, 'cnn_config.yml')
        if not os.path.exists(cnn_config_path):
            continue
        with open(cnn_config_path, 'r') as f:
            cnn_config = yaml.safe_load(f)
        entries.append(cnn_config['channels1'])
        entries.append(cnn_config['channels2'])
        entries.append(cnn_config['kernel_size'])
        entries.append(cnn_config['num_bottleneck_convs'])

        results_path = os.path.join(experiment_path, 'results.csv')
        if not os.path.exists(results_path):
            continue
        with open(results_path, 'r') as f:
            lines = f.readlines()
        line2 = lines[1].strip().split(',')
        end_accuracy = line2[0]
        entries.append(end_accuracy)
        max_accuracy = line2[2]
        entries.append(max_accuracy)

        df2 = pd.DataFrame([entries], columns=columns)
        df = df.append(df2, ignore_index=True)

    return df


