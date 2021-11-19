import os, sys
sys.path.append(os.getenv("INTRON_EXON_ROOT"))

import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from progress_bar import ProgressBar
from misc.utils import mkdir_if_not_exist

def train_epoch(inputs, labels, model, optimizer, batch_size, loss_fn):
    num_inputs = inputs.shape[0]
    indices = tf.constant(list(range(num_inputs)))
    indices_shuffled = tf.random.shuffle(indices)
    inputs_shuffled = tf.gather(inputs, indices_shuffled)
    labels_shuffled = tf.gather(labels, indices_shuffled)

    num_batches = math.ceil(num_inputs / batch_size)
    losses = np.zeros(num_batches, dtype=np.float32)

    curr_idx = 0
    i = 0
    pb = ProgressBar(num_batches)
    pb.start(front_msg="Train ")
    while curr_idx < num_inputs:
        end_idx = min(curr_idx + batch_size, num_inputs)
        input_batch = inputs_shuffled[curr_idx:end_idx]
        label_batch = labels_shuffled[curr_idx:end_idx]
        with tf.GradientTape() as tape:
            output_batch = model(input_batch)
            loss = loss_fn(output_batch, label_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses[i] = loss
        curr_idx += batch_size
        i += 1
        pb.update(front_msg="Train ")
    pb.reset()

    return np.mean(losses)

def test(inputs, labels, model, batch_size, loss_fn, metric_fn):
    num_inputs = inputs.shape[0]
    num_batches = math.ceil(num_inputs / batch_size)
    losses = np.zeros(num_batches, dtype=np.float32)
    metrics = np.zeros(num_batches, dtype=np.float32)

    curr_idx = 0
    i = 0
    pb = ProgressBar(num_batches)
    pb.start(front_msg="Val ")
    while curr_idx < num_inputs:
        end_idx = min(curr_idx + batch_size, num_inputs)
        input_batch = inputs[curr_idx:end_idx]
        label_batch = labels[curr_idx:end_idx]
        output_batch = model(input_batch)
        loss = loss_fn(output_batch, label_batch)
        metric = metric_fn(output_batch, label_batch)
        losses[i] = loss
        metrics[i] = metric
        curr_idx += batch_size
        i += 1
        pb.update(front_msg="Val ")
    pb.reset()

    return np.mean(losses), np.mean(metrics)

def checkpoint_model(model, epoch, output_dir, is_final=False):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    mkdir_if_not_exist(checkpoint_dir)
    checkpoint_filename = "checkpoint_" + ("final" if is_final else str(epoch))
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    model.save_weights(checkpoint_path)

def train_loop(train_inputs, train_labels, test_inputs, test_labels,
               model, optimizer, batch_size, loss_fn, metric_fn, metric_name,
               num_epochs, output_dir, checkpoint_freq):
    history = {'loss': [], 'val_loss': [], 'val_' + metric_name: []}

    epoch = 0
    while epoch < num_epochs:
        avg_train_loss = train_epoch(train_inputs, train_labels, model, optimizer, batch_size, loss_fn)
        avg_val_loss, avg_val_metric = test(test_inputs, test_labels, model, batch_size, loss_fn, metric_fn)

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_' + metric_name].append(avg_val_metric)
        print("[epoch %2d]  train loss: %.5f  val loss: %.5f  val %s: %.3f" %
              (epoch + 1, avg_train_loss, avg_val_loss, metric_name, avg_val_metric))

        if output_dir is not None and epoch != 0 and epoch % checkpoint_freq == 0:
            checkpoint_model(model, epoch, output_dir)

        epoch += 1

    if output_dir is not None:
        checkpoint_model(model, epoch - 1, output_dir, is_final=True)

    return history

def plot_training_history(history, metric_name=None, metric_full_name=None, save_path=None):
    loss = history['loss']
    val_loss = history['val_loss']
    train_metric, val_metric = None, None
    if metric_name is not None:
        # train_metric = history[metric_name]
        val_metric = history['val_' + metric_name]

    epochs = range(len(loss))

    plt.figure(figsize=(8, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylim([0.0, 0.7])
    plt.legend()

    if metric_name is not None:
        plt.subplot(2, 1, 2)
        # plt.plot(epochs, train_metric, 'bo', label='Training ' + metric_name)
        plt.plot(epochs, val_metric, 'b', label='Validation ' + metric_name)
        plt.title('Validation ' + metric_full_name)
        plt.ylim([0.5, 1.0])
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path)

    # plt.show()
