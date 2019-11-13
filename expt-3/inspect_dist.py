import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_2 import Input
from model_wgan import Adaptor, Mapper
from tensorflow.python.platform import app
from scipy import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.app.flags.DEFINE_string('data_file', '', 'data file')
tf.app.flags.DEFINE_integer('inspect_size', 1528, 'number of samples for testing')

def main(unused_argv):

    if FLAGS.data_file == '' or not os.path.isfile(FLAGS.data_file):
        raise ValueError('invalid data file')

    if FLAGS.output_dir == '' or not os.path.exists(FLAGS.output_dir):
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))

    checkpoint_dir = os.path.join(FLAGS.output_dir, '')

    print('reconstructing models and inputs.')
    input_ = Input(1, FLAGS.num_points)

    waves, labels = input_(FLAGS.data_file)

    adaptor = Adaptor()
    mapper = Mapper()

    logits = adaptor(waves)
    logits = mapper(logits)

    variables = snt.get_variables_in_module(adaptor) + snt.get_variables_in_module(mapper)
    saver_adaptor = tf.train.Saver(snt.get_variables_in_module(adaptor))
    saver = tf.train.Saver(variables)

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

        assert (FLAGS.inspect_size > 0), "invalid test samples"

        logits_lst = []
        labels_lst = []

        for i in range(FLAGS.inspect_size):
            logit_val, label_val = sess.run([logits, labels])

            logits_lst.append(logit_val)
            labels_lst.append(label_val)

        io.savemat("stats.mat", {"logits": logits_lst, 'classes': labels_lst})


if __name__ == '__main__':
    tf.app.run()

