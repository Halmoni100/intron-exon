import os, sys
sys.path.append(os.environ['INTRON_EXON_ROOT'])

import argparse
import yaml
import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from cnn_model import IntronExonCNN
from rnn_model import IntronExonRNN
from data.data import get_exon_ratio
from train import train_loop, plot_training_history

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', default='CNN', help='Model type')
parser.add_argument('--train_config', help='Train config file')
parser.add_argument('--model_config', help='Model config file')
parser.add_argument('--output_dir', default=None, help='Output dir for experiments')
args = parser.parse_args()

def rel_or_abs_path(path):
    if path[0] == '/':
        return path
    else:
        return os.path.join(os.environ['INTRON_EXON_ROOT'], path)

train_config_path = rel_or_abs_path(args.train_config)
with open(train_config_path, 'r') as f:
    train_config = yaml.safe_load(f)
model_config_path = rel_or_abs_path(args.model_config)
with open(model_config_path, 'r') as f:
    model_config = yaml.safe_load(f)

def get_numpy_data(window_size):
    data_dir = os.path.join(os.environ['INTRON_EXON_ROOT'], "data/split_" + str(window_size))
    train_seq_file = os.path.join(data_dir, "train_seq.npy")
    train_seq = np.load(train_seq_file)
    train_exon_file = os.path.join(data_dir, "train_exon.npy")
    train_exon = np.load(train_exon_file)
    val_seq_file = os.path.join(data_dir, "val_seq.npy")
    val_seq = np.load(val_seq_file)
    val_exon_file = os.path.join(data_dir, "val_exon.npy")
    val_exon = np.load(val_exon_file)
    test_seq_file = os.path.join(data_dir, "test_seq.npy")
    test_seq = np.load(test_seq_file)
    test_exon_file = os.path.join(data_dir, "test_exon.npy")
    test_exon = np.load(test_exon_file)
    return train_seq, train_exon, val_seq, val_exon, test_seq, test_exon

train_seq, train_exon, val_seq, val_exon, test_seq, test_exon = get_numpy_data(train_config['window_size'])
print("Exon ratio for train:", get_exon_ratio(train_exon.flatten()))
print("Exon ratio for val:", get_exon_ratio(val_exon.flatten()))

model = None
model_class = None
if args.model == 'CNN':
    model = IntronExonCNN(model_config)
    model_class = IntronExonCNN
elif args.model == 'RNN':
    model = IntronExonRNN(model_config)
    model_class = IntronExonRNN
else:
    raise ValueError('No valid model')

optimizer = tf.keras.optimizers.Adam(train_config['learning_rate'])

history = train_loop(train_seq, train_exon, val_seq, val_exon,
                     model, optimizer, train_config['batch_size'], model_class.loss, model_class.accuracy, 'acc',
                     train_config['num_epochs'], args.output_dir, 10)

val_acc = np.array(history['val_acc'])

end_acc_epoch = val_acc.shape[0] - 1
end_acc = val_acc[end_acc_epoch]
max_acc_epoch = np.argmax(val_acc)
max_acc = val_acc[max_acc_epoch]
print("End accuracy", end_acc, "at epoch", end_acc_epoch)
print("Max accuracy", max_acc, "at epoch", max_acc_epoch)

if args.output_dir is None:
    exit(0)

output_dir = rel_or_abs_path(args.output_dir)

plot_path = os.path.join(output_dir, "train_history.png")
plot_training_history(history, metric_name='acc', metric_full_name='accuracy', save_path=plot_path)

results_path = os.path.join(output_dir, "results.csv")
with open(results_path, 'w') as f:
    lines = list()
    lines.append("end_acc,end_epoch,max_acc,max_epoch\n")
    lines.append("{:.4f},{:d},{:.4f},{:d}".format(end_acc, end_acc_epoch, max_acc, max_acc_epoch))
    f.writelines(lines)
