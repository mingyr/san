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
import datetime
import math as m


class PseudoInput(snt.AbstractModule):
    def __init__(self, batch_size, z_dim, min_x = -8., max_x = 8., 
                 min_y = 0., max_y = 0.35, debug = False, name = "pseudo_input"):
        super(PseudoInput, self).__init__(name = name)

        self._z_dim = z_dim
        self._batch_size = batch_size

        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

        self._debug = False

        with self._enter_variable_scope():

            self._db_x = tf.data.Dataset.from_tensor_slices(tf.random_uniform([1, batch_size, (z_dim * 64)]))
            self._db_y = tf.data.Dataset.from_tensor_slices(tf.random_uniform([1, batch_size, (z_dim * 64)]))

            self._it_x = self._db_x.make_initializable_iterator()
            self._it_y = self._db_y.make_initializable_iterator()

            self._pi = tf.constant(m.pi, tf.float32)

    def _build(self):
        with tf.control_dependencies([self._it_x.initializer, self._it_y.initializer]):
            x = tf.identity(self._it_x.get_next())
            y = tf.identity(self._it_y.get_next())

        x = self._rescale(x, self._min_x, self._max_x)
        y = self._rescale(y, self._min_y, self._max_y)

        y_hat = self._pdf(x)

        mask = tf.where(tf.less(y, y_hat))

        x = tf.gather_nd(x, mask)
        if self._debug:
            y = tf.gather_nd(y, mask)

            return x, y
 
        outputs = tf.gather(x, range(0, self._batch_size * self._z_dim))

        outputs = tf.reshape(outputs, (self._batch_size, self._z_dim))

        return outputs


    def _pdf(self, x):
        y = 0.75 / tf.sqrt((2*self._pi)) * tf.exp(-tf.square(x - 2.0) / 2) + \
            0.25 / tf.sqrt((2*self._pi)) * tf.exp(-tf.square(x + 2.0) / 2)

        return y
            
    def _rescale(self, inputs, min_val, max_val):
        min_input = tf.reduce_min(inputs)
        max_input = tf.reduce_max(inputs)
        outputs = (inputs - min_input) * (max_val - min_val) / (max_input - min_input) + min_val
        return outputs

def test():
    pseudo_input = PseudoInput(32, 1)
    data = pseudo_input()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(10):
            value = sess.run(data)
            print(value)

        '''
        v = sess.run(data)

        print(v)
        print(v.shape)
        '''

if __name__ == "__main__":
    test()    
    # debug_test()

