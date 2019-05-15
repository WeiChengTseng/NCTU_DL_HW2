# Deep Learning (Homework 2)
> Due date : 05/17/2019
- High-level API are forbidden in this homework, such as Keras, slim, TFLearn, etc. You should implement the forward computation by yourself.
- Homework submission – Please zip each of your source code and report into a singlecompress file and name the file using this format :  HW2_StudentID_StudentName.zip(rar, 7z, tar.gz,...etc arenotacceptable)

## Problem1: Convolutional Neural Network for Image Recognition
1. Please describe in details how to preprocess the images in Animal-10 dataset with different resolutions and explain why. You have to submit your preprocessing code.  

    To create the training dataset, we use the following pre-processing technique.
    First, we apply random resizeed crop to the training images. Second, we apply random flip to the images, including vertical and horizontal flip. Third, we apply normalization to the images with mean [0.485, 0.456, 0.406] and standard devation [0.229, 0.224, 0.225].For detail implementation, please see `main_prob1.py`.

2. Please implement a CNN for image recognition. You need to design the network archi- tecture, describe your network architecture and analyze the effect of different settings including stride size and filter/kernel size. Plot learning curve, classification accuracy of training and test sets as displayed in above figure.
3. Show some examples of correctly and incorrectly classified images, list your accuracy for each test classes, and discuss about your results.

## Problem2: Recurrent Neural Network for Prediction of Paper Acceptance
1. Please construct a standard RNN for acceptance prediction.
2. Redo (1) by using the Long Short-Term Memory network (LSTM).
3. Please discuss the difference between (1) and (2).