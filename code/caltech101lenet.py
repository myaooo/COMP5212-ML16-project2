# caltech101lenet.py
# project2 - ml16
# Ming Yao

import os
import sys
import time

import numpy as np
from convnet import *
import tensorflow as tf
import scipy.io as sio

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIRECTORY = 'caltech'
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 102 # 101 categories with a background
TRAIN_SIZE = 8000
TEST_SIZE = 1144
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 100
NUM_EPOCHS = 20
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = 80  # Number of steps between evaluations.


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'caltech',
                           """Path to the Caltech101 data directory.""")

def extract_data_and_label(eval_data = False):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
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
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    labels = dict['labels']
    labels = labels.reshape(num_images)
    # reorder dimensions from [num_channel,x,y] into [x,y,num_channel]
    # data = data.transpose((0, 2, 3, 1))
    return data, labels

def main(argv=None):  # pylint: disable=unused-argument

    # Extract it into numpy arrays.
    train_data, train_labels = extract_data_and_label(eval_data=False)
    test_data, test_labels = extract_data_and_label(eval_data=True)

    num_epochs = NUM_EPOCHS

    # LeNet-5 like Model
    model = ConvNet()
    model.input_data((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), num_label=NUM_LABELS)
    model.add_conv_layer([5, 5], 32, [1, 1, 1, 1], activation='relu')
    model.add_pool('max', [1, 2, 2, 1], [1, 2, 2, 1])
    model.add_conv_layer([5, 5], 64, [1, 1, 1, 1], activation='relu')
    model.add_pool('max', [1, 2, 2, 1], [1, 2, 2, 1])
    model.add_fully_connected(512, 'relu')
    model.add_dropout(0.5)
    model.add_fully_connected(NUM_LABELS, 'relu')
    model.set_loss(tf.nn.sparse_softmax_cross_entropy_with_logits, reg=0)
    model.set_optimizer('Momentum')
    model.init()
    model.train_with_eval(train_data, train_labels, test_data, test_labels, num_epochs, EVAL_FREQUENCY)


if __name__ == '__main__':
    tf.app.run()
