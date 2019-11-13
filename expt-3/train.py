import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_2 import Input
from model_wgan import Adaptor, ReducedClassifier, Classifier
from utils import LossClassification, Summaries, Metrics, reset_metrics
from optimizer import Adam
# from optimizer import SGD
from tensorflow.python.platform import app

def train(data_path, adaptor, classifier, summ):

    input_ = Input(FLAGS.train_batch_size, FLAGS.num_points)
    waves, labels = input_(data_path)

    # Calculate the loss of the model.
    if FLAGS.adp:
        logits = tf.stop_gradient(adaptor(waves))
        # logits = adaptor(waves)
        logits = classifier(logits)
    else:
        logits = classifier(waves, expand_dims = True)

    loss = LossClassification(FLAGS.num_classes)(logits, labels)

    opt = Adam(FLAGS.learning_rate, lr_decay = True, lr_decay_steps = FLAGS.lr_decay_steps,
               lr_decay_factor = FLAGS.lr_decay_factor)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_regularization_loss = tf.reduce_sum(graph_regularizers)

    train_op = opt(loss + total_regularization_loss)

    summ.register('train', 'train_loss', loss)

    train_summ_op = summ('train')

    return loss, train_op, train_summ_op

def xval(data_path, adaptor, classifier, summ):

    input_ = Input(FLAGS.xval_batch_size, FLAGS.num_points)
    waves, labels = input_(data_path)

    # Calculate the loss of the model.
    if FLAGS.adp:
        logits = adaptor(waves)
        logits = classifier(logits)
    else:
        logits = classifier(waves, expand_dims = True)

    logits = tf.argmax(logits, axis = -1)
   
    metrics = Metrics("accuracy")
    with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
        score, xval_accu_op = metrics(labels, logits)

    assert summ, "invalid summary helper object"
    summ.register('xval', 'accuracy', score)
    xval_summ_op = summ('xval')

    return xval_accu_op, xval_summ_op

def main(unused_argv):
    summ = Summaries()

    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.data_dir))

    train_data_path = os.path.join(FLAGS.data_dir, 'eeg-train.tfr')
    xval_data_path = os.path.join(FLAGS.data_dir, 'eeg-test.tfr')

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)   

    if FLAGS.checkpoint_dir == '':
        raise ValueError('invalid checkpoint directory {}'.format(FLAGS.output_dir))

    event_log_dir = os.path.join(FLAGS.output_dir, '')
    
    checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')

    print('Constructing models.')

    if FLAGS.adp:
        adaptor = Adaptor()
        classifier = ReducedClassifier()

        train_loss, train_op, train_summ_op = \
            train(train_data_path, adaptor, classifier, summ)
        xval_op, xval_summ_op = xval(xval_data_path, adaptor, classifier, summ)
    else:
        classifier = Classifier(FLAGS.num_points, FLAGS.sampling_rate)

        train_loss, train_op, train_summ_op = \
            train(train_data_path, None, classifier, summ)
        xval_op, xval_summ_op = xval(xval_data_path, None, classifier, summ)

    print('Constructing saver.')

    if FLAGS.adp:
        variables = snt.get_variables_in_module(adaptor) + snt.get_variables_in_module(classifier)
        saver_adaptor = tf.train.Saver(snt.get_variables_in_module(adaptor))
        saver = tf.train.Saver(variables)
    else:
        saver = tf.train.Saver(snt.get_variables_in_module(classifier))

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to as some of the ops do not have GPU implementations.
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)

    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    # Build an initialization operation to run below.
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    with tf.Session(config = config) as sess:
        sess.run(init)

        if FLAGS.adp:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver_adaptor.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

        writer = tf.summary.FileWriter(event_log_dir, graph = sess.graph)

        # Run training.
        for itr in range(FLAGS.num_iterations):
            cost, _, train_summ_str = sess.run([train_loss, train_op, train_summ_op])
            # Print info: iteration #, cost.
            print(str(itr) + ' ' + str(cost))

            if itr % FLAGS.validation_interval == 1:
                # Run through validation set.
                sess.run(xval_op)
                val_summ_str = sess.run(xval_summ_op)
                writer.add_summary(val_summ_str, itr)
                reset_metrics(sess)
            
            if itr % FLAGS.summary_interval == 1:
                writer.add_summary(train_summ_str, itr)

        # coord.request_stop()
        # coord.join(threads)

        tf.logging.info('Saving model.')
        saver.save(sess, checkpoint_path)
        tf.logging.info('Training complete')

if __name__ == '__main__':
    app.run()

