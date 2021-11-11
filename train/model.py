import tensorflow as tf

class IntronExonCNN(tf.keras.Model):
    def __init__(self, channels1, channels2, kernel_size, num_bottleneck_convs):
        super(IntronExonCNN, self).__init__()

        self.E_nucleotide = tf.keras.layers.Embedding(4, 4)

        self.max_pooling = tf.keras.layers.MaxPooling1D(2)
        self.conv_f1 = tf.keras.layers.Conv1D(channels1, kernel_size, activation='relu', padding='same')
        self.conv_f2 = tf.keras.layers.Conv1D(channels2, kernel_size, activation='relu', padding='same')

        if num_bottleneck_convs % 2 != 0:
            raise ValueError("Need even number of bottleneck convolutions for residual connections")

        self.bottleneck_convs = []
        for _ in range(num_bottleneck_convs):
            self.bottleneck_convs.append(tf.keras.layers.Conv1D(channels2, kernel_size, activation='relu', padding='same'))

        self.up_sampling = tf.keras.layers.UpSampling1D(2)
        self.conv_b1 = tf.keras.layers.Conv1D(channels2, kernel_size, activation='relu', padding='same')
        self.conv_b2 = tf.keras.layers.Conv1D(channels1, kernel_size, activation='relu', padding='same')

        self.final = tf.keras.layers.Conv1D(1, kernel_size, activation='sigmoid', padding='same')

    def call(self, x):
        x = self.E_nucleotide(x)

        x = self.conv_f1(x)
        x = self.max_pooling(x)
        x = self.conv_f2(x)
        x = self.max_pooling(x)

        for i in range(0, len(self.bottleneck_convs), 2):
            x1 = self.bottleneck_convs[i](x)
            x2 = self.bottleneck_convs[i+1](x1)
            x += x2

        x = self.conv_b1(x)
        x = self.up_sampling(x)
        x = self.conv_b2(x)
        x = self.up_sampling(x)
        output = self.final(x)
        return output


