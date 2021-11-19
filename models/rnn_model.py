import tensorflow as tf
import numpy as np

class IntronExonRNN(tf.keras.Model):
    def __init__(self, rnn_config):
        super(IntronExonRNN, self).__init__()

        self.E_nucleotide = tf.keras.layers.Embedding(4, 4)

        self.stack = list()
        for _ in range(rnn_config['num_stacks']):
            self.stack.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)))

        self.final = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.E_nucleotide(x)

        for bidirectional_lstm in self.stack:
            x = bidirectional_lstm(x)

        output = self.final(x)
        return output

    @staticmethod
    def loss(outputs, labels):
        losses = tf.keras.metrics.binary_crossentropy(labels, outputs, from_logits=False)
        return tf.reduce_mean(losses)

    @staticmethod
    def accuracy(outputs, labels, threshold=0.5):
        outputs_gt_threshold = np.where(outputs > threshold, 1, 0)
        m = tf.keras.metrics.binary_accuracy(labels, outputs_gt_threshold)
        return np.sum(m) / (m.shape[0] * m.shape[1])