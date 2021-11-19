import tensorflow as tf
import numpy as np

class Hybrid(tf.keras.Model):
    def __init__(self, config):
        super(Hybrid, self).__init__()

        self.E_nucleotide = tf.keras.layers.Embedding(4, 4)

        self.conv_f1 = tf.keras.layers.Conv1D(32, 10, activation='relu', padding='same')
        self.conv_f2 = tf.keras.layers.Conv1D(32, 10, strides=2, activation='relu', padding='same')
        self.conv_f3 = tf.keras.layers.Conv1D(32, 10, strides=2, activation='relu', padding='same')
        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))
        self.conv_b3 = tf.keras.layers.Conv1DTranspose(16, 10, strides=2, activation='relu', padding='same')
        self.conv_b2 = tf.keras.layers.Conv1DTranspose(8, 10, strides=2, activation='relu', padding='same')
        self.final = tf.keras.layers.Conv1D(1, 10, activation='sigmoid', padding='same')

    def call(self, x):
        x = self.E_nucleotide(x)

        x = self.conv_f1(x)
        x = self.conv_f2(x)
        x = self.conv_f3(x)

        x = self.rnn(x)

        x = self.conv_b3(x)
        x = self.conv_b2(x)
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