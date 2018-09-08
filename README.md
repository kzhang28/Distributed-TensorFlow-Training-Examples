# Project Description
Examples for distributed training of machine learning/deep learning models in TensorFlow. Every model training example can be run on a multi-node cluster. 

# Contents
This repository contains a few examples for distributed (multi-nodes) training on Tensorflow (test on CPU cluster)  
1. Single layer neural network: [mnist_nn_distibuted_placeholder.py](https://github.com/kzhang28/tensorflow_example/blob/master/mnist_nn_distibuted_placeholder.py)
2. Softmax model: [mnist_softmax_distibuted_placeholder.py](https://github.com/kzhang28/tensorflow_example/blob/master/mnist_softmax_distibuted_placeholder.py)
3. Two hidden layers neural network: [mnist_2hiddenLayerNN_distributed_ph.py](https://github.com/kzhang28/tensorflow_example/blob/master/mnist_2hiddenLayerNN_distributed_ph.py)
4. CNN tensorflow example 


# Usage
1. For model 1,2,3: you can find a script called `xxx.py` and a corresponding folder 
in which there are shell scripts to launch the distributed training job. 
2. For model 4: please refer to the corresponding [README](https://github.com/kzhang28/Distributed-TensorFlow-Training-Examples/blob/master/Alexnet/README.md)
# Note:
- Change some default setting (e.g., python path, HOME path, host name) before running each training job.
- Make sure you understand the basics of distributed Tensorflow. See the [offical tutorial](https://www.tensorflow.org/deploy/distributed) for more detail.

# Version and Environment
- Model 1,2,3: Tensorflow version: 0.11.0rc0, Python 3, Ubuntu 16
- Model 4: Tensorflow version 1.5.0, Python 3, Ubuntu 16

