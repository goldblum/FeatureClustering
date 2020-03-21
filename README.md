# FeatureClustering

This repository contains PyTorch code for the feature clustering and hyperplane variation regularizers from ["Unraveling Meta-Learning: Understanding Feature Representations for Few-Shot Tasks
"](https://arxiv.org/abs/2002.06753) by Micah Goldblum, Steven Reich, Liam Fowl, Renkun Ni, Valeriia Cherepanova, and Tom Goldstein.

The feature clustering and hyperplane variation regularizers enforce clustering in feature space in order to encourage better few-shot performance in the transfer learning setting for classically trained (non-meta-learned) models.  We show in our paper that these regularizers have a similar effect to meta-learning on feature extractors.  This work is not intended to achieve state-of-the-art performance but to instead develop a better understanding of how meta-learning works.

This repository supports the mini-ImageNet and CIFAR-FS datasets as well as a variety of backbone architectures.  Please download datasets independently in order to run the code.

## Prerequisites
* Python3
* PyTorch
* CUDA

## Run
Here is an example of how to run our program:
```
$ python train_feature_extractor.py
```
