# Joint Distribution across Representation Spaces for Out-of-Distribution Detection

This repository contains the essential code for the paper [Joint Distribution across Representation Spaces for Out-of-Distribution Detection](https://arxiv.org/).

## Preliminaries

It is tested under Ubuntu Linux 16.04 and Python 3.8 environment, and requries Pytorch package to be installed:

* [PyTorch](http://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [scikit-learn](http://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)

## Out-of-Distribtion Datasets

We use the following out-of-distributin datasets, we provide the links here:

* [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

## Pre-trained Models

Some pre-trained neural networks are the same as [deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector). They can be downloaded from:

* [DenseNet on CIFAR-10](https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth?dl=0) / [DenseNet on CIFAR-100](https://www.dropbox.com/s/7ur9qo81u30od36/densenet_cifar100.pth?dl=0)
* [ResNet on CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0) / [ResNet on CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0)

The others are contained in this repository.

## How to use

```bash
# model: DenseNet, in-distribution: CIFAR-100, batch_size: 200
python test_lsgm.py --method_name densenet_cifar100 --test_bs 200
```
