import numpy as np
import tensorflow as tf
import cv2
import sys


class PNet(tf.keras.Model):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = conv2d(10, 3, 1, 'valid', name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(name='prelu1', shared_axes=[1, 2])
        self.pool1 = pool2d(name='pool1')

        self.conv2 = conv2d(16, 3, 1, 'valid', name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(name='prelu2', shared_axes=[1, 2])

        self.conv3 = conv2d(32, 3, 1, 'valid', name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(name='prelu3', shared_axes=[1, 2])

        self.conv4_1 = conv2d(
            2, 1, 1, 'same', activation='softmax', name='conv4_1')
        self.conv4_2 = conv2d(4, 1, 1, 'same', name='conv4_2')
        self.model_variable_initialize()

    def call(self, input_):
        out = self.conv1(input_)
        out = self.prelu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        out = self.conv3(out)
        out = self.prelu3(out)

        prob = self.conv4_1(out)
        loc = self.conv4_2(out)
        return prob, loc

    def model_variable_initialize(self):
        image = tf.random_normal((1, 12, 12, 3))
        with tf.name_scope('PNet'):
            self.call(image)
        print("PNet variables initialize completed")

    def restore(self):
        self.load_weights("./checkpoints/pnet/model")


class RNet(tf.keras.Model):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = conv2d(28, 3, 1, 'valid', name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(name='prelu1', shared_axes=[1, 2])
        self.pool1 = pool2d(3, 2, name='pool1')

        self.conv2 = conv2d(48, 3, 1, 'valid', name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(name='prelu2', shared_axes=[1, 2])
        self.pool2 = pool2d(3, 2, 'valid', name='pool2')

        self.conv3 = conv2d(64, 2, 1, 'valid', name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(name='prelu3', shared_axes=[1, 2])

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, name='fc1')
        self.prelu4 = tf.keras.layers.PReLU(name='prelu4')

        self.fc2_1 = tf.keras.layers.Dense(
            2, activation='softmax', name='fc2_1')
        self.fc2_2 = tf.keras.layers.Dense(4, name='fc2_2')
        self.model_variable_initialize()

    def call(self, input_):
        out = self.conv1(input_)
        out = self.prelu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.prelu3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.prelu4(out)

        prob = self.fc2_1(out)
        loc = self.fc2_2(out)
        return prob, loc

    def model_variable_initialize(self):
        image = tf.random_normal((1, 24, 24, 3))
        with tf.name_scope('RNet'):
            self.call(image)
        print("RNet variables initialize completed")

    def restore(self):
        self.load_weights("./checkpoints/rnet/model")


class ONet(tf.keras.Model):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = conv2d(32, 3, 1, 'valid', name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(name='prelu1', shared_axes=[1, 2])
        self.pool1 = pool2d(3, 2, name='pool1')

        self.conv2 = conv2d(64, 3, 1, 'valid', name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(name='prelu2', shared_axes=[1, 2])
        self.pool2 = pool2d(3, 2, 'valid', name='pool2')

        self.conv3 = conv2d(64, 3, 1, 'valid', name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(name='prelu3', shared_axes=[1, 2])
        self.pool3 = pool2d(2, 2, name='pool3')

        self.conv4 = conv2d(128, 2, 1, 'valid', name='conv4')
        self.prelu4 = tf.keras.layers.PReLU(name='prelu4', shared_axes=[1, 2])

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, name='fc1')
        self.prelu5 = tf.keras.layers.PReLU(name='prelu5')

        self.fc2_1 = tf.keras.layers.Dense(
            2, activation='softmax', name='fc2_1')
        self.fc2_2 = tf.keras.layers.Dense(4, name='fc2_2')
        self.fc2_3 = tf.keras.layers.Dense(10, name='fc2_3')
        self.model_variable_initialize()

    def call(self, input_):
        out = self.conv1(input_)
        out = self.prelu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.prelu3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.prelu4(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.prelu5(out)

        prob = self.fc2_1(out)
        loc = self.fc2_2(out)
        landmark = self.fc2_3(out)
        return prob, loc, landmark

    def model_variable_initialize(self):
        image = tf.random_normal((1, 48, 48, 3))
        with tf.name_scope('ONet'):
            self.call(image)
        print("ONet variables initialize completed")

    def restore(self):
        self.load_weights("./checkpoints/onet/model")


def conv2d(filter, ksize=3, stride=1, padding='same', dilation=1, activation=None, name="conv2d"):
    ksize = [ksize] * 2
    strides = [stride] * 2
    dilation = [dilation] * 2
    return tf.keras.layers.Conv2D(filters=filter, kernel_size=ksize, strides=strides, padding=padding, dilation_rate=dilation, activation=activation, name=name)


def pool2d(ksize=2, stride=2, padding='same', name='pool2d'):
    ksize = [ksize] * 2
    strides = [stride]*2
    return tf.keras.layers.MaxPool2D(pool_size=ksize, strides=strides, padding=padding, name=name)
