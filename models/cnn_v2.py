import tensorflow as tf
import numpy as np

class CNNV2(tf.keras.Model):
    def __init__(self, cnn_v2_config):
        super(CNNV2, self).__init__()

        stride = cnn_v2_config['stride']
        dilation = cnn_v2_config['dilation']

        self.E_nucleotide = tf.keras.layers.Embedding(4, 4)

        self.max_pooling = tf.keras.layers.MaxPooling1D(2)

        self.conv_f1 = tf.keras.layers.Conv1D(32, 10, activation='relu', padding='same')
        self.conv_f2 = tf.keras.layers.Conv1D(16, 10, strides=stride, dilation_rate=dilation, activation='relu', padding='same')
        self.conv_f3 = tf.keras.layers.Conv1D(16, 10, strides=stride, dilation_rate=dilation, activation='relu', padding='same')
        self.conv_f4 = tf.keras.layers.Conv1D(16, 10, strides=stride, dilation_rate=dilation, activation='relu', padding='same')
        self.conv_b4 = tf.keras.layers.Conv1DTranspose(16, 10, strides=stride, activation='relu', padding='same')
        self.conv_b3 = tf.keras.layers.Conv1DTranspose(16, 10, strides=stride, activation='relu', padding='same')
        self.conv_b2 = tf.keras.layers.Conv1DTranspose(32, 10, strides=stride, activation='relu', padding='same')
        self.final = tf.keras.layers.Conv1D(1, 10, activation='sigmoid', padding='same')

    def call(self, x):
        x = self.E_nucleotide(x)

        x = self.conv_f1(x)
        x = self.conv_f2(x)
        x = self.conv_f3(x)
        x = self.conv_f4(x)

        x = self.conv_b4(x)
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