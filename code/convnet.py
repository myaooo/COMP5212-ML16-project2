#
# project: project 2
# file: convnet.py
# author: MING Yao
#

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import standard_ops
import math
import time
import sys

SEED = None
EVAL_BATCH_SIZE = 100

def activation(str = 'linear'):
    if str == 'linear':
        return lambda x: x
    elif str == 'relu':
        return tf.nn.relu
    else:
        return tf.nn.relu

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

class ConvNet:
    """A wrapper class for conv net in tensorflow"""

    def __init__(self):
        self.layers = []

    def input_data(self, dshape=None, num_label=2, dtype=tf.float32):
        """
        :param shape: (batch_size, size_x, size_y, num_channel)
        :param dtype: data type
        :return:
        """
        self.dshape = dshape
        self.num_channel = dshape[3]
        self.dtype = dtype
        self.num_label = num_label

        # creating placeholder
        self.train_data_node = tf.placeholder(dtype, dshape)
        self.train_labels_node = tf.placeholder(tf.int64, shape=(dshape[0],))
        self.eval_data = tf.placeholder(dtype, shape=(EVAL_BATCH_SIZE, dshape[1], dshape[2],dshape[3]))
        self.layers.append({'type':'input', 'output':list(dshape[1:])})

    def add_conv_layer(self, filter, depth, strides, padding = 'SAME', activation = 'linear', bias = True):
        """
        filter: should be a (x,y) double
        depth: the depth (number of filter) within the layer
        strides: a list of int indicating filter strides, should be a list of 4 int
        padding: padding algorithm, could be 'SAME' or 'VALID'
        activation: tensorflow activation type default to linear, frequent use include 'relu'
        """

        assert len(strides) == 4
        self.layers.append({'type': 'conv', 'filter': filter, 'depth': depth, 'strides': strides,
                            'padding': padding, 'activation': activation, 'isbias': bias})


    def add_pool(self, type, kernel_size, strides, padding = 'SAME'):
        """

        :param type: type of pooling, could be 'max' or 'avg'
        :param kernel_size: a list of int indicating kernel size
        :param strides:
        :param padding:
        :return:
        """
        assert len(strides) == 4
        self.layers.append({'type': type, 'kernel': kernel_size, 'strides': strides, 'padding': padding})

    def add_fully_connected(self, n_units, activation='linear', bias=True):
        self.layers.append({'type':'fully_connected', 'depth': n_units, 'activation': activation, 'isbias': bias})


    def add_dropout(self, prob):
        self.layers.append({"type":"dropout",'prob':prob})


    def set_loss(self, loss, reg = 0):
        if isinstance(loss, type(tf.nn.softmax)):
            self.loss_function = loss
        else:
            self.loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits

        self.regular_coef = reg

    def set_optimizer(self, str):
        self.optimizer=None
        if str == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif str == 'GradientDescent':
            self.optimizer = tf.train.GradientDescentOptimizer
        elif str == 'Adam':
            self.optimizer = tf.train.AdamOptimizer
        elif str == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer
        elif str == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer
        else:
            self.optimizer = tf.train.MomentumOptimizer


    def init(self):
        """Call this before training"""

        # initialize network layers' parameters
        dtype = self.dtype
        layers = self.layers
        for n, layer in enumerate(layers):
            # don't need to initialized input layer
            if layer['type'] == 'max' or layer['type'] == 'min':
                output_ = layers[n - 1]['output']
                strides = layer['strides']
                x = math.ceil(output_[0] / strides[1])
                y = math.ceil(output_[1] / strides[2])
                out_channel = layers[n-1]['depth']
                layer['output'] = [x, y, out_channel]
            # conv layer
            elif layer['type'] == 'conv':
                output_ = layers[n-1]['output']
                strides = layer['strides']
                x = math.ceil(output_[0]/strides[1])
                y = math.ceil(output_[1]/strides[2])
                in_channel = output_[2]
                out_channel = layer['depth']
                layer['output'] = [x,y,out_channel]
                layer['shape'] = [layer['filter'][0], layer['filter'][1], in_channel, out_channel]
                layer['weight'] = tf.Variable(tf.truncated_normal(layer['shape'], stddev=0.1, seed=SEED, dtype=dtype))
                if layer['isbias']:
                    layer['bias'] = tf.Variable(tf.constant(0.1, shape=[out_channel], dtype=dtype))

            # fully connected layer
            elif layer['type'] == 'fully_connected':
                output_ = layers[n - 1]['output']
                if len(output_) > 1:
                    layer['shape'] = [output_[0]*output_[1]*output_[2],layer['depth']]
                else:
                    layer['shape'] = [output_[0], layer['depth']]
                layer['output'] = [layer['depth']]
                layer['weight'] = tf.Variable(tf.truncated_normal(layer['shape'], stddev=0.1, seed=SEED, dtype=dtype))
                if layer['isbias']:
                    layer['bias'] = tf.Variable(tf.constant(0.1, shape=[layer['depth']], dtype=dtype))
            elif layer['type'] == 'dropout' or layer['type'] == 'input':
                continue
            else:
                print('Initializing: Unknown layer!')

        # initialize loss and optimizers
        self.logits = self.model(self.train_data_node, True)
        self.loss = tf.reduce_mean(self.loss_function(self.logits, self.train_labels_node))
        self.loss += self.regularizer()
        # Predictions for the current training minibatch.
        self.train_prediction = tf.nn.softmax(self.logits)

        # Predictions for the test and validation, which we'll compute less often.
        self.eval_prediction = tf.nn.softmax(self.model(self.eval_data))


    def model(self, data, train = False):
        """The Model definition."""

        # Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        layers = self.layers
        result = data
        status = 'conv'
        for n, layer in enumerate(layers):
            if layer['type'] == 'input':
                continue
            if layer['type'] == 'conv':
                result = tf.nn.conv2d(result, layer['weight'], layer['strides'], layer['padding'])
                if layer['isbias']:
                    result = activation(layer['activation'])(tf.nn.bias_add(result, layer['bias']))
            elif layer['type'] == 'max':
                result = tf.nn.max_pool(result, layer['kernel'], layer['strides'], layer['padding'])
            elif layer['type'] == 'min':
                result = tf.nn.avg_pool(result, layer['kernel'], layer['strides'], layer['padding'])
            elif layer['type'] == 'dropout' and train:
                result = tf.nn.dropout(result, layer['prob'], seed=SEED)
            elif layer['type'] == 'fully_connected':
                if status == 'conv':
                    status = 'fc'
                    # reshape the result
                    shape = result.get_shape().as_list()
                    result = tf.reshape(result, [shape[0], shape[1] * shape[2] * shape[3]])
                result = tf.matmul(result, layer['weight'])
                if layer['isbias']:
                    result = result + layer['bias']
                if n == len(layers)-1:
                    # return logit value, no need to call activation function
                    break
                else:
                    # call activation function
                    result = activation(layer['activation'])(result)
        return result


    def train_with_eval(self, train_data, train_labels, test_data, test_labels, num_epochs=20, eval_frequency = 100):
        """training function"""

        loss = self.loss
        # logits = self.logits
        train_size = train_labels.shape[0]

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch_size = self.dshape[0]
        batch_size = train_size if (batch_size == None) else batch_size

        batch = tf.Variable(0, dtype=self.dtype)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch_size*batch,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = self.optimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

        # Create a local session to run the training.
        with tf.Session() as sess:

            total_time = time.time()
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()
            # Initialize logging arrays
            epoch = []
            train_accuracy = []
            test_accuracy = []
            train_loss = []
            test_loss = []
            time_length = []
            print('Initialized!')
            # Loop through training steps.
            start_time = time.time()

            for step in range(int(num_epochs * train_size) // batch_size + 1):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * batch_size) % (train_size - batch_size)
                batch_data = train_data[offset:(offset + batch_size), ...]
                batch_labels = train_labels[offset:(offset + batch_size)]
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph it should be fed to.
                feed_dict = {self.train_data_node: batch_data,
                             self.train_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                _, lr = sess.run(
                    [optimizer, learning_rate],
                    feed_dict=feed_dict)
                if step % (eval_frequency / 10) == 0:
                    print('Step %d (epoch %.2f)' %
                          (step, float(step) * batch_size / train_size))
                if step % eval_frequency == 0 and step != 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    epoch.append(float(step) * batch_size / train_size)
                    time_length.append(elapsed_time)
                    # Evaluate
                    train_error = error_rate(self.eval_in_batches(train_data, sess), train_labels)
                    train_accuracy.append(1 - 0.01 * train_error)
                    test_error = error_rate(self.eval_in_batches(test_data, sess), test_labels)
                    test_accuracy.append(1 - 0.01 * test_error)
                    train_loss.append(self.current_loss(train_data,train_labels,sess))
                    test_loss.append(self.current_loss(test_data,test_labels,sess))
                    print('Epoch %.2f, %.1f ms per step' %
                          (epoch[-1], 1000 * elapsed_time / eval_frequency))
                    print('Mean train loss: %.3f, mean test loss: %.3f, learning rate: %.6f'
                          % (train_loss[-1], test_loss[-1], lr))
                    print('Train accuracy: %.1f%%' % (100 * train_accuracy[-1]))
                    print('Test accuracy: %.1f%%' % (100 * test_accuracy[-1]))
                    sys.stdout.flush()

        total_time = time.time() - total_time
        print('Total running time: %.2f s' % total_time)
        current_str = time.strftime("%m%d%H%M",time.localtime())
        np.savetxt('./output/out'+current_str+'.csv', [epoch, time_length, train_loss, train_accuracy, test_accuracy], delimiter=',')

    def regularizer(self):
        if self.regular_coef == 0:
            return 0
        regular = tf.Variable(0,dtype=self.dtype)
        for layer in self.layers:
            if layer['type'] == 'fully_connected':
                regular += tf.nn.l2_loss(layer['weight']) + tf.nn.l2_loss(layer['bias'])
        return self.regular_coef * regular

    def eval_in_batches(self, data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, self.num_label), dtype=np.float32)
        for begin in range(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    self.eval_prediction,
                    feed_dict={self.eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    self.eval_prediction,
                    feed_dict={self.eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def current_loss(self, data, labels, sess):

        size = data.shape[0]
        batch_size = self.dshape[0]
        r = range(0,size,batch_size)
        n = len(r)
        loss = np.ndarray(shape=(n),dtype=np.float32)
        for i, begin in enumerate(r):
            end = begin + batch_size
            loss[i] = sess.run(self.loss, feed_dict={
                self.train_data_node: data[begin:end, ...], self.train_labels_node: labels[begin:end, ...]})
        return loss.sum() / n
