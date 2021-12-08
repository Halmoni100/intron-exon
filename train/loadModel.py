import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

import argparse
import yaml,math
import tensorflow as tf
import numpy as np
from itertools import groupby

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from models.cnn_model import IntronExonCNN
from models.rnn_model import IntronExonRNN
from models.cnn_rnn_hybrid import Hybrid
from models.cnn_v2 import CNNV2
from data.data import get_exon_ratio
from train import train_loop, plot_training_history, test
from progress_bar import ProgressBar

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', default='CNN', help='Model type')
parser.add_argument('--train_config', help='Train config file')
parser.add_argument('--model_config', help='Model config file')
parser.add_argument('--output_dir', default=None, help='Output dir for experiments')
args = parser.parse_args()

with open(args.train_config, 'r') as f:
    train_config = yaml.safe_load(f)
with open(args.model_config, 'r') as f:
    model_config = yaml.safe_load(f)

def get_numpy_data(window_size):
    data_dir = os.path.join(os.getenv("INTRON_EXON_ROOT"), "data", "split_" + str(window_size))
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



def countExons(vec):
    return sum([(vec[i:i+2]==[False,True]).all() for i in range(len(vec)-1)])+(1 if vec[0]==True else 0)

def getLength(vec):
    tmp = np.array([[smth,sum(1 for _ in group)] for smth, group in groupby(vec)])
    return tmp[tmp[:,0]==1,1],tmp[tmp[:,0]==0,1]


def eval(inputs, labels, model, batch_size, loss_fn, metric_fn):
    num_inputs = inputs.shape[0]
    num_batches = math.ceil(num_inputs / batch_size)
    # save evaluations
    losses = np.zeros(num_batches, dtype=np.float32)
    metrics = np.zeros(num_batches, dtype=np.float32)
    nExon_pred = np.zeros((num_batches,batch_size), dtype=np.float32)
    lenExon = []
    lenIntron = []

    curr_idx = 0
    i = 0
    pb = ProgressBar(num_batches)
    pb.start(front_msg="Val ")
    while curr_idx < num_inputs:
        end_idx = min(curr_idx + batch_size, num_inputs)
        input_batch = inputs[curr_idx:end_idx]
        label_batch = labels[curr_idx:end_idx]
        # predict
        output_batch = model(input_batch)
        # loss
        losses[i] = loss_fn(output_batch, label_batch)
        metrics[i]  = metric_fn(output_batch, label_batch)
        # number of exons
        output_batch = output_batch.numpy()
        output_batch = output_batch.reshape((output_batch.shape[0:2]))
        tmp=[countExons(vec>.5) for vec in output_batch]
        nExon_pred[i,:len(tmp)] = tmp
        # length of exon introns
        for b in output_batch:
            tmp = getLength(b>.5)
            lenExon.append(tmp[0])
            lenIntron.append(tmp[1])

        curr_idx += batch_size
        i += 1
        pb.update(front_msg="Val ")
    pb.reset()

    return nExon_pred,lenExon,lenIntron, np.mean(losses), np.mean(metrics)



# get data
train_seq, train_exon, val_seq, val_exon, test_seq, test_exon = get_numpy_data(train_config['window_size'])
print("Exon ratio for train:", get_exon_ratio(train_exon.flatten()))
print("Exon ratio for val:", get_exon_ratio(val_exon.flatten()))


# set up model
model = None
model_class = None
if args.model == 'CNN':
    model = IntronExonCNN(model_config)
    model_class = IntronExonCNN
elif args.model == 'CNNV2':
    model = CNNV2(model_config)
    model_class = CNNV2
elif args.model == 'RNN':
    model = IntronExonRNN(model_config)
    model_class = IntronExonRNN
elif args.model == 'HYBRID':
    model = Hybrid(model_config)
    model_class = Hybrid
else:
    raise ValueError('No valid model')

# load model weights
checkpoint_dir = "experiments\cnn\experiment_10\checkpoints\checkpoint_final"
model.load_weights(checkpoint_dir)
##########print("total number of parameters")
#############print(model.count_params())
#_,acc_train = test(train_seq, train_exon, model, train_config['batch_size'], model_class.loss, model_class.accuracy)
#_, acc_val = test( val_seq, val_exon, model, train_config['batch_size'], model_class.loss, model_class.accuracy)

val_nExon_pred,val_lenExon,val_lenIntron,val_loss,val_accuracy = eval(val_seq, val_exon,model, train_config['batch_size'], model_class.loss, model_class.accuracy)
train_nExon_pred,train_train_lenExon,train_lenIntron,train_loss,train_accuracy = eval(train_seq, train_exon,model, train_config['batch_size'], model_class.loss, model_class.accuracy)
nPar = model.count_params()

print(nExon_pred,lenExon,lenIntron)
