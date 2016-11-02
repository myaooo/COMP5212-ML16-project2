% Machine Learning 2016 Fall
  \break Project 2
% Ming Yao (# 20365468)
% October 17th, 2016

# Introduction

In this project, experimental study on two types of CNN models (LeNet and Network in Network) 
and three real-world data sets (MNIST, CIFAR-10, Caltech 101) is conducted.

Due to several reasons, such as my laptop issues and MATLAB installing issues on server, experiments are implemented based on TensorFlow[^TS]. 
All the experiments are run on a server with the following environment:

* OS: Debian GNU/Linux 8.5 (Jessie)

* CPU: Intel(R) Core(TM) i7-4930K CPU @ 3.40GHz (6 cores, 12 processors)

* GPU: None

For extendability and convenience, I abstract a simple `ConvNet`[^convnet] class that wrap a few TensorFlow operations in modeling a convolution network. 
`ConvNet` offers convenient operations like `add_*_layer` for model definition and computation. 
For detail code please refer to `code/convnet.py`. 

The structure of the project is shown as below:

\dirtree{%
    .1 project2/. 
    .2 code/. 
    .3 caltech101lenet.py. 
    .3 caltech101nin.py.
    .3 cifar10.py.
    .3 convnet.py. 
    .3 mnist\_new.py. 
    .3 run.sh. 
    .3 caltech/. 
    .3 cifar-10/. 
    .3 data/. 
    .3 output/.
}

[^TS]: Detail documentations are at [`http://www.tensorflow.org`](http://www.tensorflow.org)

[^convnet]: The `ConvNet` class is written with reference of the example model TensorFlow offered in [`tensorflow/models/image`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/image).


# LeNet model for MNIST and CIFAR-10

## MNIST

## CIFAR-10


# Training CNN Models on Caltech 101

## LeNet Model

## Network in Network Model

# Conclusion