"""
This is a straightforward Python implementation of a generative adversarial network.
The code is drawn directly from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

A version of this model with explanatory notes is also available on GitHub
at https://github.com/jonbruner/generative-adversarial-networks.

This script requires TensorFlow and its dependencies in order to run. Please see
the readme for guidance on installing TensorFlow.

This script won't print summary statistics in the terminal during training;
track progress and see sample images in TensorBoard.
"""

import sonnet as snt
import numpy as np
import tensorflow as tf
import sonnet as snt
from utils import Activation, Downsample1D, spectral_norm
import datetime

# Define the discriminator network
class Discriminator(snt.AbstractModule):
    def __init__(self, act = 'elu', name = "discriminator"):
        super(Discriminator, self).__init__(name = name)

        self._act = Activation(act, verbose = True)

        with self._enter_variable_scope():
            self._d_w1 = tf.get_variable('d_w1', [1, 256], initializer = tf.truncated_normal_initializer(stddev = 1))
            self._d_b1 = tf.get_variable('d_b1', [256], initializer = tf.constant_initializer(0))

            self._d_w2 = tf.get_variable('d_w2', [256, 64], initializer = tf.truncated_normal_initializer(stddev = 1))
            self._d_b2 = tf.get_variable('d_b2', [64], initializer = tf.constant_initializer(0))

            self._d_w3 = tf.get_variable('d_w3', [64, 1], initializer = tf.truncated_normal_initializer(stddev = 1))
            self._d_b3 = tf.get_variable('d_b3', [1], initializer = tf.constant_initializer(0))

    def _build(self, inputs):
        outputs = tf.identity(inputs)

        outputs = tf.matmul(inputs, spectral_norm(self._d_w1))
        outputs = outputs + self._d_b1
        outputs = self._act(outputs)

        outputs = tf.matmul(outputs, spectral_norm(self._d_w2))
        outputs = outputs + self._d_b2
        outputs = self._act(outputs)

        outputs = tf.matmul(outputs, spectral_norm(self._d_w3))
        outputs = outputs + self._d_b3

        return outputs


class Adaptor(snt.AbstractModule):
    def __init__(self, num_filters = 32, filter_size = 5, 
                 act = '', name = "adaptor"):
        super(Adaptor, self).__init__(name = name)

        self._bf = snt.BatchFlatten()
        self._pool = Downsample1D(2)
        self._act = Activation(act, verbose = True)

        with self._enter_variable_scope():

            self._l1_conv = snt.Conv1D(num_filters, filter_size + 2)
            self._l2_conv = snt.Conv1D(num_filters << 1, filter_size)
            self._l3_conv = snt.Conv1D(num_filters << 2, filter_size - 2)

    def _build(self, inputs):
        outputs = tf.expand_dims(inputs, axis = -1)

        outputs = self._l1_conv(outputs)
        outputs = self._act(outputs)
        outputs = self._pool(outputs)

        outputs = self._l2_conv(outputs)
        outputs = self._act(outputs)
        outputs = self._pool(outputs)

        outputs = self._l3_conv(outputs)
        outputs = self._pool(outputs)

        return outputs


class Mapper(snt.AbstractModule):
    def __init__(self, filter_size = 5, num_filters = 32,
                 pooling_stride = 2, act = 'tanh', summ = None, name = "mapper"):
        super(Mapper, self).__init__(name = name)
        
        self._pool = Downsample1D(pooling_stride)
        self._act = Activation(act, verbose = True)
        self._bf = snt.BatchFlatten()

        self._summ = summ

        initializers = {
            'w': tf.truncated_normal_initializer(stddev = 0.01),
            'b': tf.zeros_initializer()
        }

        with self._enter_variable_scope():
            self._lin1 = snt.Linear(64, initializers = initializers)
            self._lin2 = snt.Linear(1, initializers = initializers)

    def _build(self, inputs):
        outputs = self._bf(inputs)

        outputs = self._lin1(outputs)
        outputs = self._act(outputs)

        outputs = self._lin2(outputs)

        if self._summ: self._summ.register('train', 'real_dist', outputs)

        return outputs


class ReducedClassifier(snt.AbstractModule):
    def __init__(self, num_classes = 2, act = 'elu', name = "reduced_classifier"):
        super(ReducedClassifier, self).__init__(name = name)

        self._act = Activation(act, verbose = True)
        self._bf = snt.BatchFlatten()

        regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                        "b": tf.contrib.layers.l2_regularizer(scale=0.1)}

        with self._enter_variable_scope():
            self._lin1 = snt.Linear(256, regularizers = regularizers)
            self._lin2 = snt.Linear(num_classes, regularizers = regularizers)

    def _build(self, inputs, expand_dims = True):
        outputs = self._bf(inputs)

        outputs = self._lin1(outputs)
        outputs = self._act(outputs)

        outputs = self._lin2(outputs)

        return outputs


class Classifier(snt.AbstractModule):
    def __init__(self, sampling_rate, filter_size = 3, num_filters = 32,
                 pooling_stride = 2, pool = 'avg', act = 'elu', name = "classifier"):
        super(Classifier, self).__init__(name = name)

        num_classes = 2

        self._act = Activation(act, verbose = True)
        self._pool = Downsample1D(2) 
        self._bf = snt.BatchFlatten()

        regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=0.1),
                        "b": tf.contrib.layers.l2_regularizer(scale=0.1)}

        with self._enter_variable_scope():
            self._l1_conv = snt.Conv1D(num_filters, filter_size + 2)
            self._l2_sepconv = snt.SeparableConv1D(num_filters << 1, 1, filter_size)
            self._lin1 = snt.Linear(256, regularizers = regularizers)
            self._lin2 = snt.Linear(num_classes, regularizers = regularizers)

    def _build(self, inputs, expand_dims = True):
        if expand_dims:
            outputs = tf.expand_dims(inputs, axis = -1)
        else:
            outputs = inputs

        outputs = self._l1_conv(outputs)
        outputs = self._act(outputs)
        outputs = self._pool(outputs)

        outputs = self._l2_sepconv(outputs)
        outputs = self._act(outputs)
        outputs = self._pool(outputs)
        
        outputs = self._bf(outputs)
        
        outputs = self._lin1(outputs)
        outputs = self._act(outputs)

        outputs = self._lin2(outputs)

        return outputs


def test_adaptor():
    print("test adaptor")

    from config import FLAGS
    from input_2 import Input
    input_ = Input(32, FLAGS.num_points)
    adaptor = Adaptor(FLAGS.num_filters)

    waves, labels = input_('/data/yuming/eeg-processed-data/vep/san-1d/eeg.tfr') 
    outputs = adaptor(waves)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(outputs)

        print(v.shape)


def test_mapper():
    print("test generator")

    from config import FLAGS
    from input_2 import Input
    input_ = Input(32, FLAGS.num_points)

    adaptor = Adaptor(FLAGS.num_filters)
    mapper = Mapper()

    waves, labels = input_('/data/yuming/eeg-processed-data/vep/san-1d/eeg.tfr')
    outputs = adaptor(waves)
    outputs = mapper(outputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(outputs)

        print(v.shape)


def test_discriminator():
    print("test discriminator")

    from input_3 import PseudoInput

    input_ = PseudoInput(32, 1)
    discriminator = Discriminator()

    outputs = discriminator(input_())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(outputs)

        print(v.shape)

def test_classifier():
    print("test classifier")

    from config import FLAGS

    from input_2 import Input
    input_ = Input(32, FLAGS.num_points)

    waves, labels = input_('/data/yuming/eeg-processed-data/vep/san-1d/eeg.tfr')

    classifier = Classifier(FLAGS.num_points, FLAGS.sampling_rate)

    outputs = classifier(waves)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(outputs)

        print(v.shape)

if __name__ == '__main__':

    test_adaptor()

    # test_mapper()

    # test_discriminator()

    # test_classifier()


