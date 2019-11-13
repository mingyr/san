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


def prepare(filename, test_subjects = None, is_training = True):

    test_subjects = [int(s) for s in test_subjects.split(',')]

    if len(test_subjects) == 0:
        raise ValueError("invalid test subjects")

    print('test subject identifiers are {}'.format(test_subjects))

    content = sio.loadmat(filename)
    content = content['new_subjects']
    content = np.squeeze(content)

    all_set = set(range(len(content)))
    
    for test_subject in test_subjects:
        assert test_subject in all_set, "invalid test subject"

    train_set = all_set - set(test_subjects)

    data = []
    labels = []

    if is_training:
        for i in train_set:

            print("processing subject {} for training".format(i))

            datum = content[i][0]
            label = content[i][1]

            for j in range(datum.shape[-1]):
                data.append(datum[30, :, j].flatten())
                labels.append(label[j] if label[j][0] > 0 else [0])

        assert (len(data) == len(labels)), "mismatched data and labels quantity"
      
        total = len(labels)
        indices = range(total)
        shuffle(indices)

        pivot = int(total * 0.85)

        print("total samples {}".format(total))
        print("training samples {}".format(pivot))
        print("validation samples {}".format(total - pivot))

        data_train = []
        labels_train = []

        data_xval = []
        labels_xval = []

        for i in range(total):
            if i < pivot:
                data_train.append(data[indices[i]])
                labels_train.append(labels[indices[i]])
            else:
                data_xval.append(data[indices[i]]) 
                labels_xval.append(labels[indices[i]])

        data_train = np.array(data_train)
        labels_train = np.array(labels_train)
        labels_train = labels_train.astype(np.int64)

        data_xval = np.array(data_xval)
        labels_xval = np.array(labels_xval)
        labels_xval = labels_xval.astype(np.int64)

        tfrec_filename_train = os.path.join(FLAGS.outdir, 'eeg-train.tfr')
        writer_train = tf.python_io.TFRecordWriter(tfrec_filename_train)

        total = len(labels_train)
        for t in range(total):
            print("currently processing training trial no. {} / total training trials {}".format(t + 1, total))

            wave = data_train[t]
            label = labels_train[t]

            gen_tfr(wave, label, writer_train)

        writer_train.close()

        tfrec_filename_xval = os.path.join(FLAGS.outdir, 'eeg-xval.tfr')
        writer_xval = tf.python_io.TFRecordWriter(tfrec_filename_xval)

        total = len(labels_xval)
        for t in range(total):
            print("currently processing validation trial no. {} / total validation trials {}".format(t + 1, total))

            wave = data_xval[t]
            label = labels_xval[t]

            gen_tfr(wave, label, writer_xval)

        writer_xval.close()
                
    else:

        for i in test_subjects:

            print("processing subject {} for training".format(i))

            datum = content[i][0]
            label = content[i][1]

            for j in range(datum.shape[-1]):
                data.append(datum[30, :, j].flatten())
                labels.append(label[j] if label[j][0] > 0 else [0])

        assert (len(data) == len(labels)), "mismatched data and labels quantity"

        total = len(labels)
        indices = range(total)
        shuffle(indices)

        print("testing samples {}".format(total))

        data_test = []
        labels_test = []

        for i in range(total):
            data_test.append(data[indices[i]])
            labels_test.append(labels[indices[i]])

        data_test = np.array(data)
        labels_test = np.array(labels)
        labels_test = labels_test.astype(np.int64)

        tfrec_filename_test = os.path.join(FLAGS.outdir, 'eeg-test.tfr')
        writer_test = tf.python_io.TFRecordWriter(tfrec_filename_test)

        for t in range(total):
            print("currently processing test trial no. {} / total test trials {}".format(t + 1, total))

            wave = data_test[t]
            label = labels_test[t]

            gen_tfr(wave, label, writer_test)

        writer_test.close()

 
def main(unused_argv):

    if not os.path.exists(FLAGS.outdir):
        os.makedirs(FLAGS.outdir)
        print("store tfrecords files in directory {}".format(FLAGS.outdir))

    filename = 'vep_data.mat'

    test_subjects = '4'

    prepare(filename, test_subjects)
    prepare(filename, test_subjects, False)

if __name__ == '__main__':
    app.run()


