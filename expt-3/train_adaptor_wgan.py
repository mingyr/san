import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_1 import PseudoInput
from input_2 import Input
from model_wgan import Discriminator, Adaptor, Mapper
from utils import Summaries
from tensorflow.python.platform import app


def train(data_path, summ):

    input_target = PseudoInput(FLAGS.batch_size, 1)
    input_source = Input(FLAGS.batch_size, FLAGS.num_points)

    adaptor = Adaptor()
    mapper = Mapper(summ = summ)
    discriminator = Discriminator()

    inputs, _ = input_source(data_path) 
    
    # Calculate the loss of the model.

    feat = adaptor(inputs)
    feat = mapper(feat)

    target = input_target()
    summ.register('train', 'target_dist', target)

    logits_source = discriminator(feat)
    logits_target = discriminator(target)

    # WGAN Loss
    d_loss_real = -tf.reduce_mean(logits_target)
    d_loss_fake = tf.reduce_mean(logits_source)
    d_loss = d_loss_real + d_loss_fake

    # Total generator loss.
    g_loss = -d_loss_fake

    summ.register("train", "Discriminator_loss", d_loss)
    summ.register("train", "Generator_loss", g_loss)

    generator_vars = snt.get_variables_in_module(adaptor) + snt.get_variables_in_module(mapper)
    discriminator_vars = snt.get_variables_in_module(discriminator)

    g_optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.adp_learning_rate, name = 'g_opt',
        beta1 = FLAGS.beta1, beta2 = FLAGS.beta2).minimize(g_loss, var_list = generator_vars)

    d_optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.adp_learning_rate, name = 'd_opt',
        beta1 = FLAGS.beta1, beta2 = FLAGS.beta2).minimize(d_loss, var_list = discriminator_vars)

    summ_op = summ('train')

    return g_optimizer, d_optimizer, summ_op

def main(unused_argv):
    summ = Summaries()

    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.data_dir))

    data_path = os.path.join(FLAGS.data_dir, 'eeg.tfr')

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)   

    event_log_dir = os.path.join(FLAGS.output_dir, '')
    checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')

    g_op, d_op, summ_op = train(data_path, summ)

    print('Constructing saver.')
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to as some of the ops do not have GPU implementations.
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)

    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    # Build an initialization operation to run below.
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    with tf.Session(config = config) as sess:
        sess.run(init)

        writer = tf.summary.FileWriter(event_log_dir, graph = sess.graph)

        # Run training.
        for itr in range(FLAGS.num_adapt_iterations):
            sess.run(d_op)

            # Print info: iteration #, cost.
            # print(str(itr) + ' ' + str(cost))

            sess.run(g_op)

            if itr % FLAGS.summary_interval == 1:
                summ_str = sess.run(summ_op)
                writer.add_summary(summ_str, itr)

        tf.logging.info('Saving model.')
        saver.save(sess, checkpoint_path)
        tf.logging.info('Training complete')

if __name__ == '__main__':
    app.run()

