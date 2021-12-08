import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

import argparse
import yaml,math,csv,pickle
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



##################################
# get path to each experiment
directory = "experiments"
expList = []
for entry in os.scandir(directory):
    for entry1 in os.scandir(entry):
        expList.append(entry1.path)#print(ent.path)

# get data
train_seq, train_exon, val_seq, val_exon, test_seq, test_exon = get_numpy_data(1000) # train_config['window_size'] ??? window size is always 1000
print("Exon ratio for train:", get_exon_ratio(train_exon.flatten()))
print("Exon ratio for val:", get_exon_ratio(val_exon.flatten()))

AllResultFile = open(directory+"\\results1_all.csv", 'w', newline='')
AllResultFileWriter = csv.writer(AllResultFile)
AllResultFileWriter.writerow(["experiment","nPar","val_loss","val_accuracy","val_nExon_median","val_lenExon_median","val_lenIntron_median","train_loss","train_accuracy","train_nExon_median","train_lenExon_median","train_lenIntron_median","model_config","train_config"])

# for each experiment
for experiment in expList:
    print(experiment)
    # load config
    with open(experiment+"\\train_config.yml", 'r') as f:
        train_config = yaml.safe_load(f)
    for ent in os.scandir(experiment):
        if (ent.path.endswith(".yml") and not ent.path.endswith("train_config.yml")):
            with open(ent.path, 'r') as f:
                model_config = yaml.safe_load(f)
    # path to saved weights
    pathToSavedWeights = experiment+"\checkpoints\checkpoint_final"

    if "\\cnn\\" in experiment:
        model = IntronExonCNN(model_config)
        model_class = IntronExonCNN
    elif  "\\cnn_v2" in experiment:
        model = CNNV2(model_config)
        model_class = CNNV2
    elif  "\\hybrid\\" in experiment:
        model = Hybrid(model_config)
        model_class = Hybrid
    elif  "\\rnn\\" in experiment:
        model = IntronExonRNN(model_config)
        model_class = IntronExonRNN
    else:
        raise ValueError('No valid model')

    # load model weights
    model.load_weights(pathToSavedWeights)
    # evaluate model
    val_nExon_pred,val_lenExon,val_lenIntron,val_loss,val_accuracy = eval(val_seq, val_exon,model, train_config['batch_size'], model_class.loss, model_class.accuracy)
    train_nExon_pred,train_lenExon,train_lenIntron,train_loss,train_accuracy = eval(train_seq, train_exon,model, train_config['batch_size'], model_class.loss, model_class.accuracy)

    nPar = model.count_params()
    val_nExon_median = np.median(val_nExon_pred[val_nExon_pred!=0])
    val_lenExon_median = np.median(np.concatenate(val_lenExon,axis=None))
    val_lenIntron_median = np.median(np.concatenate(val_lenIntron,axis=None))
    train_nExon_median = np.median(train_nExon_pred[train_nExon_pred!=0])
    train_lenExon_median = np.median(np.concatenate(train_lenExon,axis=None))
    train_lenIntron_median = np.median(np.concatenate(train_lenIntron,axis=None))

    results_path =experiment+"\\results1.csv"
    resultFile = open(results_path, 'w', newline='')
    resultFileWriter = csv.writer(resultFile)
    resultFileWriter.writerow(["experiment","nPar","val_loss","val_accuracy","val_nExon_median","val_lenExon_median","val_lenIntron_median","train_loss","train_accuracy","train_nExon_median","train_lenExon_median","train_lenIntron_median","model_config","train_config"])
    resultFileWriter.writerow([experiment,nPar,val_loss,val_accuracy,val_nExon_median,val_lenExon_median,val_lenIntron_median,train_loss,train_accuracy,train_nExon_median,train_lenExon_median,train_lenIntron_median,model_config,train_config])
    resultFile.close()
    # Saving the objects:
    resultFile =  open(experiment+'\\resultObjects.pkl', 'wb')
    pickle.dump([val_nExon_pred,val_lenExon,val_lenIntron,train_nExon_pred,train_lenExon,train_lenIntron], resultFile)
    resultFile.close()
    # write to universal file
    AllResultFileWriter.writerow([experiment,nPar,val_loss,val_accuracy,val_nExon_median,val_lenExon_median,val_lenIntron_median,train_loss,train_accuracy,train_nExon_median,train_lenExon_median,train_lenIntron_median,model_config,train_config])


AllResultFile.close()
