# Project Description
Three distributed training examples in Tensorflow.  

# Contents
This repository contain three examples/models for handwriting digit recognition (MNIST dataset)
1. Single layer neural network: [mnist_nn_distibuted_placeholder.py](https://github.com/kzhang28/tensorflow_example/blob/master/mnist_nn_distibuted_placeholder.py)
2. Softmax model:[mnist_softmax_distibuted_placeholder.py](https://github.com/kzhang28/tensorflow_example/blob/master/mnist_softmax_distibuted_placeholder.py)
3. Two hidden layers neural network: [mnist_2hiddenLayerNN_distributed_ph.py](https://github.com/kzhang28/tensorflow_example/blob/master/mnist_2hiddenLayerNN_distributed_ph.py)

I run the three models for resoure utilization comparison. If you have more interest in distributed machine learning platforms you can see [my paper.](http://www.eden.rutgers.edu/~kz181/ICCCN.pdf)
# Usage
1. For each example `xxx.py`,
you can find the corresponding folder 
in which there are scripts to run it in distributed mode. Change the python path and HOME path before run it.

# Version
Tensorflow version: 0.11.0rc0
