import os, sys
sys.path.append(os.environ['INTRON_EXON_ROOT'])

import argparse
import yaml
import tensorflow as tf

from cnn_model import IntronExonCNN
from data.data import initial_process_data, preprocess
from train import train_loop

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', default='CNN', help='Model type')
parser.add_argument('--train_config', help='Train config file')
parser.add_argument('--model_config', help='Model config file')
args = parser.parse_args()

train_config_path = os.path.join(os.environ['INTRON_EXON_ROOT'], args.train_config)
train_config = yaml.load(train_config_path)
model_config_path = os.path.join(os.environ['INTRON_EXON_ROOT'], args.model_config)
model_config = yaml.load(model_config_path)

seq_train, seq_dict_train, exon_coord_train = initial_process_data(train_config['seq_train_file'],
                                                                   train_config['exon_train_file'])
windowed_seq_train, windowed_exon_train = preprocess(seq_train, exon_coord_train, train_config['window_size'])

seq_test, seq_dict_test, exon_coord_test = initial_process_data(train_config['seq_test_file'],
                                                                train_config['exon_test_file'])
windowed_seq_test, windowed_exon_test = preprocess(seq_test, exon_coord_test, train_config['window_size'])

model = None
model_class = None
if args.model == 'CNN':
    model = IntronExonCNN(model_config)
    model_class = IntronExonCNN
else:
    raise ValueError('No valid model')

optimizer = tf.keras.optimizers.Adam(train_config['learning_rate'])

train_loop(windowed_seq_train, windowed_exon_train, windowed_seq_test, windowed_exon_test,
           model, optimizer, train_config['batch_size'], model_class.loss, model_class.accuracy, 'acc',
           train_config['num_epochs'])



