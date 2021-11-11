import math

import tensorflow as tf
import numpy as np

from progress_bar import ProgressBar

def train_epoch(inputs, labels, model, optimizer, batch_size, loss_fn):
    num_inputs = inputs.shape[0]
    indices = tf.constant(list(range(num_inputs)))
    indices_shuffled = tf.random.shuffle(indices)
    inputs_shuffled = tf.gather(inputs, indices_shuffled)
    labels_shuffled = tf.gather(labels, indices_shuffled)

    num_batches = math.ceil(num_inputs / model.batch_size)
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
        end_idx = min(curr_idx + model.batch_size, num_inputs)
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

def train_loop(train_inputs, train_labels, test_inputs, test_labels,
               model, optimizer, batch_size, loss_fn, metric_fn, metric_name,
               num_epochs):
    history = {'loss': [], 'val_loss': [], 'val_' + metric_name: []}

    epoch = 0
    while epoch > num_epochs:
        avg_train_loss = train_epoch(train_inputs, train_labels, model, optimizer, batch_size, loss_fn)
        avg_val_loss, avg_val_metric = test(test_inputs, test_labels, model, batch_size, metric_fn)

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_' + metric_name].append(avg_val_metric)
        print("[epoch %2d]  train loss: %.5f  val loss: %.5f  val %s: %.3f" %
              (epoch + 1, avg_train_loss, avg_val_loss, metric_name, avg_val_metric))

        epoch += 1


