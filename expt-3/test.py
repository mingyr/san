import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_2 import Input
from model_wgan import Adaptor, ReducedClassifier, Classifier
from utils import Metrics
from tensorflow.python.platform import app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(unused_argv):

    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory')

    if FLAGS.evaluate:
        print("evaluate the model")
        data_path = os.path.join(FLAGS.data_dir, 'eeg-xval.tfr')
    else:
        print("model inference")
        data_path = os.path.join(FLAGS.data_dir, 'eeg-test.tfr')

    if FLAGS.output_dir == '' or not os.path.exists(FLAGS.output_dir):
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))

    checkpoint_dir = os.path.join(FLAGS.output_dir, '')

    print('reconstructing models and inputs.')
    input_ = Input(1, FLAGS.num_points)

    waves, labels = input_(data_path)

    if FLAGS.adp:
        adaptor = Adaptor()
        classifier = ReducedClassifier()

        logits = adaptor(waves)
        logits = classifier(logits)
    else:

        classifier = Classifier(FLAGS.num_points, FLAGS.sampling_rate)
        logits = classifier(waves, expand_dims = True)

    # Calculate the loss of the model.
    logits = tf.argmax(logits, axis = -1)
    
    metrics = Metrics("accuracy")
    with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
        metric_op, metric_update_op = metrics(labels, logits)
   
    if FLAGS.adp:
        variables = snt.get_variables_in_module(adaptor) + snt.get_variables_in_module(classifier)
        saver_adaptor = tf.train.Saver(snt.get_variables_in_module(adaptor))
        saver = tf.train.Saver(variables)
    else:
        saver = tf.train.Saver(snt.get_variables_in_module(classifier))

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        assert (FLAGS.test_size > 0), "invalid test samples"
        for i in range(FLAGS.test_size):
            sess.run(metric_update_op)

        metric = sess.run(metric_op)
        print("metric -> {}".format(metric))

if __name__ == '__main__':
    tf.app.run()

