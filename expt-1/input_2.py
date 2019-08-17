'''
@author: MSUser
'''
import os
import numpy as np
import tensorflow as tf
import sonnet as snt 

class Input(snt.AbstractModule):
    def __init__(self, batch_size, data_shape = None,
                 num_epochs = -1, name = 'input'):
        '''
        Args:
            batch_size: number of tfrecords to dequeue
            data_shape: the expected shape of series of images
            num_enqueuing_threads: enqueuing threads
        '''
        super(Input, self).__init__(name = name)

        self._batch_size = batch_size
        self._data_shape = data_shape
        self._num_epochs = num_epochs

        assert data_shape, "invalid data shape" 

    def _parse_function(self, example):
        dims = np.prod(self._data_shape)

        features = {
            "image": tf.FixedLenFeature([dims], dtype = tf.float32),
            "label": tf.FixedLenFeature([], dtype = tf.int64),
        }

        example_parsed = tf.parse_single_example(serialized = example,
                                                 features = features)
		
        image = tf.reshape(example_parsed['image'], self._data_shape) 

        return image, example_parsed['label']

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
        images, labels = iterator.get_next()

        return images, labels
    
if __name__ == '__main__':

    input_ = Input(64, [28, 28])
    images, labels = input_('/data/yuming/eeg-processed-data/mnist/mnist.tfr')
	
    with tf.Session() as sess:

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for i in range(10):        
            images_val, labels_val = sess.run([images, labels])

            # np.set_printoptions(threshold = np.nan)

            print(images_val.shape)

            print(labels_val)


