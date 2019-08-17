# Copyright 2017 Yurui Ming (yrming@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for configuring the prediction model."""

import tensorflow as tf
from tensorflow.python.platform import flags

# might contains dangling configuration entries

# begin of MNIST data specification

flags.DEFINE_integer('img_height', 28, 'number of classes to classify.')
flags.DEFINE_integer('img_width', 28, 'number of classes to classify.')

flags.DEFINE_integer('num_classes', 2, 'number of classes to classify.')

flags.DEFINE_integer('num_filters', 32, 'number of filters')
flags.DEFINE_integer('pooling_stride', 2, 'pooling stride')

flags.DEFINE_integer('num_adapt_iterations', 20000, 'number of training iterations.')
flags.DEFINE_integer('num_iterations', 1000, 'number of training iterations.')

# warning: to mapping a large domain into a small range, just pickup a tiny learning rate
flags.DEFINE_float('learning_rate', 0.0001, 'the base learning rate of the generator')

flags.DEFINE_float('lr_decay_factor', 0.90, 'learning rate decay factor')
flags.DEFINE_float('lr_decay_steps', 100, 'after the specified steps then learning rate decay')
flags.DEFINE_integer('num_epochs_per_decay', 10, 'after how many epochs shall the learning rate decay.')

flags.DEFINE_string('metric', 'f1_score', 'specific metric to measure model performance')

flags.DEFINE_integer('test_size', 412, 'number of samples for testing')

flags.DEFINE_float('adp_learning_rate', 0.0001, 'the base learning rate of the generator')
flags.DEFINE_float('beta1', 0.0, 'beta 1')
flags.DEFINE_float('beta2', 0.9, 'beta 2')

# end of MNIST data specification

# general configurations below

flags.DEFINE_string('gpus', '', 'visible GPU list')
flags.DEFINE_string('type', '', 'type of the model, classification or regression')

flags.DEFINE_string('data_dir', '', 'directory of data')
flags.DEFINE_string('output_dir', '', 'directory for model outputs.')
flags.DEFINE_string('checkpoint_dir', '', 'directory of checkpoint files.')

# subject adaptation
flags.DEFINE_boolean('adp', True, 'adaptation or not')

flags.DEFINE_integer('batch_size', 64, 'batch size for training.')

flags.DEFINE_integer('train_batch_size', 0, 'batch size for training.')
flags.DEFINE_integer('xval_batch_size', 0, 'batch size for cross evaluation')
flags.DEFINE_integer('test_batch_size', 0, 'batch size for test')

flags.DEFINE_integer('summary_interval', 5, 'how often to record tensorboard summaries.')
flags.DEFINE_integer('validation_interval', 10, 'how often to run a batch through the validation model')
# flags.DEFINE_integer('save_interval', 2000, 'how often to save a model checkpoint.')

flags.DEFINE_boolean('validation', True, 'whether do validation or not')
flags.DEFINE_boolean('evaluate', True, 'evaluate using validation data to select model parameters')

flags.DEFINE_integer('sample_size', 0, 'sample size for analyzing the model')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


