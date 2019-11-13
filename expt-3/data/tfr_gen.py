'''
'''

import os, sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from random import shuffle

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

flags.DEFINE_string('outdir', '', 'output directory')
FLAGS = flags.FLAGS

# list file candidates

os.path.exists(FLAGS.outdir), "invalid output directory: {}".format(FLAGS.outdir)

# index in python is zero based
num_chs = 1
num_pts = 45

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def gen_tfr(wave, label, writer):
    feature = {
        'wave': float_feature(np.reshape(wave, [-1])),
        'class': int64_feature(label),
    }

    example = tf.train.Example(features = tf.train.Features(feature = feature))
    writer.write(example.SerializeToString())


def prepare(filename):

    content = sio.loadmat(filename)
    content = content['new_subjects']
    content = np.squeeze(content)

    data = []
    labels = []

    for i in range(len(content)):

        datum = content[i][0]
        label = content[i][1]

        for j in range(datum.shape[-1]):
            data.append(datum[30, :, j].flatten())
            labels.append(label[j] if label[j][0] > 0 else [0])

    assert (len(data) == len(labels)), "mismatched data and labels quantity"
      
    total = len(labels)
    indices = range(total)
    shuffle(indices)

    print("total samples {}".format(total))

    data = np.array(data)
    labels = np.array(labels)

    data = data[indices, ...]
    labels = labels[indices, ...]

    # data = np.array(data)
    # labels = np.array(labels)
    labels = labels.astype(np.int64)

    tfrec_filename = os.path.join(FLAGS.outdir, 'eeg.tfr')
    writer = tf.python_io.TFRecordWriter(tfrec_filename)

    for t in range(total):
        print("currently processing trial no. {} / total trials {}".format(t + 1, total))

        wave = data[t]
        label = labels[t]

        gen_tfr(wave, label, writer)

    writer.close()

 
def main(unused_argv):

    if not os.path.exists(FLAGS.outdir):
        os.makedirs(FLAGS.outdir)
        print("store tfrecords files in directory {}".format(FLAGS.outdir))

    filename = 'vep_data.mat'

    prepare(filename)

if __name__ == '__main__':
    app.run()


