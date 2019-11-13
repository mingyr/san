'''
@author: MSUser
'''
import os
import numpy as np
import tensorflow as tf
import sonnet as snt 

class Input(snt.AbstractModule):
    def __init__(self, batch_size, num_points,
                 num_epochs = -1, name = 'input'):
        '''
        Args:
            batch_size: number of tfrecords to dequeue
            data_shape: the expected shape of series of images
            num_enqueuing_threads: enqueuing threads
        '''
        super(Input, self).__init__(name = name)

        self._batch_size = batch_size
        self._num_points = num_points
        self._num_epochs = num_epochs

    def _parse_function(self, example):

        features = {
            "wave": tf.FixedLenFeature([self._num_points], dtype = tf.float32),
            "class": tf.FixedLenFeature([], dtype = tf.int64),
        }

        example_parsed = tf.parse_single_example(serialized = example,
                                                 features = features)
		
        return example_parsed['wave'], example_parsed['class']

    def _build(self, filenames):
        '''
        Retrieve tfrecord from files and prepare for batching dequeue
        Args:
            filenames: 
        Returns:
            wave label in batch
        '''

        assert os.path.isfile(filenames), "invalid file path: {}".format(filenames)
		
        if type(filenames) == list:
            dataset = tf.data.TFRecordDataset(filenames)
        elif type(filenames) == str:
            dataset = tf.data.TFRecordDataset([filenames])
        else:
            raise ValueError('wrong type {}'.format(type(filenames)))

        dataset = dataset.map(self._parse_function)

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(self._num_epochs)

        iterator = dataset.make_one_shot_iterator()
        waves, labels = iterator.get_next()

        return waves, labels
    
if __name__ == '__main__':

    input_ = Input(128, 45)
    waves, labels = input_('/data/yuming/eeg-processed-data/vep/san-new/eeg.tfr')
	
    with tf.Session() as sess:

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        
        waves_val, labels_val = sess.run([waves, labels])

        # np.set_printoptions(threshold = np.nan)

        print(waves_val.shape)
        print(labels_val)


