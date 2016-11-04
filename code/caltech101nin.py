# caltech101nin.py
# project2 - ml16
# Ming Yao

import os

import numpy as np
from convnet import *
import tensorflow as tf
import scipy.io as sio
from utils import *

DATA_DIRECTORY = 'caltech'
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 102 # 101 categories with a background
TRAIN_SIZE = 8000
TEST_SIZE = 1144
BATCH_SIZE = 50
NUM_EPOCHS = 100
EVAL_FREQUENCY = int(TRAIN_SIZE/BATCH_SIZE)  # Number of steps between evaluations.


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'caltech',
                           """Path to the Caltech101 data directory.""")

def extract_data_and_label(eval_data = False):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are normalized
    """
    data_list = []
    label_list = []
    num_images = TRAIN_SIZE
    filename = os.path.join(DATA_DIRECTORY, 'data_batch_1.mat')
    if eval_data:
        num_images = TEST_SIZE
        filename = os.path.join(DATA_DIRECTORY, 'test_batch.mat')

    print('Extracting', filename)
    dict = sio.loadmat(filename)
    # parsing data
    data = dict['data']
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = normalize_data(data)
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    labels = dict['labels']
    labels = labels.reshape(num_images)

    return data, labels

def CompleteZCAwhitening(train_data, test_data):
    train_data = train_data.reshape(TRAIN_SIZE, IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)
    test_data = test_data.reshape(TEST_SIZE, IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)
    data = ZCAwhitening(np.vstack([train_data,test_data]))
    train_data = data[:TRAIN_SIZE,:]
    test_data = data[TRAIN_SIZE:,:]
    train_data = train_data.reshape(TRAIN_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    test_data = test_data.reshape(TEST_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    
    return train_data, test_data

def main(argv=None):  # pylint: disable=unused-argument

     # Extract it into numpy arrays.
    train_data, train_labels = extract_data_and_label(eval_data=False)
    test_data, test_labels = extract_data_and_label(eval_data=True)
    train_data, test_data = CompleteZCAwhitening(train_data,test_data)

    num_epochs = NUM_EPOCHS

    # Network in Network
    model = ConvNet()
    model.input_data((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), num_label=NUM_LABELS, eval_batch=BATCH_SIZE)
    model.add_conv_layer(filter=[5, 5], depth=192, strides=[1, 1, 1, 1], activation='relu')
    model.add_conv_layer(filter=[1, 1], depth=160, strides=[1, 1, 1, 1], activation='relu')
    model.add_conv_layer(filter=[1, 1], depth=96, strides=[1, 1, 1, 1], activation='relu')
    model.add_pool('max', kernel_size=[1, 3, 3, 1], strides=[1, 2, 2, 1])
    model.add_dropout(0.5)
    model.add_conv_layer(filter=[5, 5], depth=192, strides=[1, 1, 1, 1], activation='relu')
    model.add_conv_layer(filter=[1, 1], depth=192, strides=[1, 1, 1, 1], activation='relu')
    model.add_conv_layer(filter=[1, 1], depth=192, strides=[1, 1, 1, 1], activation='relu')
    model.add_pool('max', kernel_size=[1, 3, 3, 1], strides=[1, 2, 2, 1])
    model.add_dropout(0.5)
    model.add_conv_layer(filter=[3, 3], depth=192, strides=[1, 1, 1, 1], activation='relu')
    model.add_conv_layer(filter=[1, 1], depth=192, strides=[1, 1, 1, 1], activation='relu')
    model.add_conv_layer(filter=[1, 1], depth=NUM_LABELS, strides=[1, 1, 1, 1], activation='relu')
    model.add_pool('avg', kernel_size=[1, 8, 8, 1], strides=[1, 8, 8, 1])
    model.add_flatten()
    model.set_loss(tf.nn.sparse_softmax_cross_entropy_with_logits, reg=0)
    model.set_optimizer('Adam')
    model.init()
    model.train_with_eval(train_data, train_labels, test_data, test_labels, num_epochs, EVAL_FREQUENCY, 0.001)


if __name__ == '__main__':
    tf.app.run()
