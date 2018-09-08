# Description
 - This is a distributed version of Tensorflow Advanced Convolutional Neural Networks Example: [Tensorflow CNN example tutorial](https://www.tensorflow.org/tutorials/images/deep_cnn). The goal of the original tutorial aims at building a convolutional neural network (Alexnet with a few differences in the top few layers). 
 - The code of the original example can be found [here](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/).
 - This distributed training example is based on the original [code](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/) and makes a few modification (e.g., adding some training hooks) to facilitate distributed training.
# Contents
1. `cifar10_distributed.py` --contains training entry point
2. `cifar10_input.py`  --contains utility functions for handling data input
# Usage
Please refer to [distributed tensorflow](https://www.tensorflow.org/deploy/distributed). You need at least specify the following arguments:
`--ps_hosts, --worker_hosts, --job_name, --task_index`
# Note
- `cifar10_distributed.py` and `cifar10_input.py` should be placed in the same directory when launching each training task.
- Modify default arugment value and global constant in `cifar10_distributed.py` as needed.
#  Runtime Environment
- Tensorflow 1.5.0+
- Python 3
- Ubuntu 16


