# Fiera: Swift as a Wild beast

<p align="center">
<img src="https://raw.githubusercontent.com/xyzunreal/Fiera/master/images/fiera.png" width = 650 height = 300/>
</p>


Binarised Neural Networks will play a huge role in bringing AI to the edge devices. Fiera aims to be user-friendly and highly optimized binary neural network framework for studying and implementing BNNs.

## Features

* Binarize both inputs and weights. Currenly we support binarization of both inputs and weights but later we want to give user flexibility to choose.
* Supports both forward and backward propogation. Most of the frameworks present today are only inference based which cannot leverage benefits of research focused on training strategies to increase accuracy and training speed of BNNs specifically.
* Use c++ or either will use libraries of c++ which will prevent from slow function calls of high level languages and also allow to write assembly level optimizations.
* Solely built on C++ that can benefit from machine level optimizations.
* Currenly we are only binarizing Convolution layers but later we will also allow to binarize Fully Connected(FC) layers if user want.
* We provide support for binarization of both fc and conv layers.


Note: Our current version is providing PROOF OF CONCEPT of training and inference of  Binary Neural Networks. We have provided guidelines to test it yourself on MNIST.

## Binarization

Input image and weights are both `float` initially. Both images and weights are binarized using `sign` function.
Binarized inputs and weights are channel-wise packed into `int64`. That's why we currently need channels to be multiple of 64.
Apply normal convolution operation but instead of multiplying input and weights, use xnor operation.
Take `2*p-n` where `p` is result of previous step and `n` is size of input.

## Usage

To test framework on MNIST dataset.

```

git clone https://github.com/xyzunreal/Fiera
cd Fiera/binary_cnn/Example\ MNIST/full_test
g++ --std=c++11 -o mnist MNIST.cpp
./mnist

```

You can train model yourself by changing in MNIST.cpp

## Todo

#### Speed and Accuracy:

* Replace simple convolutions with im2col convolutions. Later we will also consider using Winograd and FFTs.
* Leverage CPU cache properly by using proper memory layout for tensors.
* Use shift-based batch normalization.
* Flexible channel Support -  We currently support layers with 64 channels only which works fine for many new architectures where we have multiples of 64 as number of channels. This makes packing bits channelwise possible in `int64`.  But to make it more generalized, add padding of zeroes to make incoming channels multiple of 64 and change 2p-n formula accordingly.

#### DevOps:

* Unit Tests and Integration Tests (using GoogleTest)
* CMake
* Logging (glog)
* Continous Integration

#### Features and Design:

* Provide Cuda support.
* Use callbacks wherever possible to make framework more flexible to implement new modules.
* Implement autograd. Binarized layers will still use Straight Through Estimator.
* Implement different weight initializers, loss functions, optimizers etc.
* ONNX support to increase interoperability between Fiera and other frameworks.
* Allow users to download Datasets and Trained models from fiera.io. Provide pretrained models of high speed models like MobileNet, Squeezenet first.
* Add Documentation on [Fiera.io](fiera.io) and improve Github Documentation .

#### Later Down the Road

* Other Quantization levels (2-bit, 4-bit, 8-bit).
* Other techniques for network compression like Network Pruning.

## Support

Join chat at [Gitter](https://https://gitter.im/Fiera-UnrealAI/)
 
## Why Fiera?

Fiera means 'Wild Beast' in Spanish. It reflects Swifness, Agility and Robustness of Fiera.
