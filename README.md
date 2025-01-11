# Iris DNN From Scratch

Deep neural network for Iris classification implemented from scratch using NumPy only. No ML frameworks used.

[Colab For DNN](https://colab.research.google.com/drive/1gbArpy5Oy0RGYPrAepD5FcR81HeDmYXz?usp=sharing) 

## Overview
This project implements a deep neural network to classify Iris flowers into three species. The entire neural network is built from scratch using only NumPy, demonstrating the fundamental concepts of deep learning.

## Features
- Custom implementation of:
  - Sigmoid activation function
  - Softmax output layer
  - Cross-entropy loss
  - Backpropagation
- Data preprocessing and splitting (70% train, 30% test)
- Model evaluation metrics

## Requirements
python
numpy==1.21.0

## Model Architecture
- Input layer: 4 neurons (sepal length, sepal width, petal length, petal width)
- Hidden layer: 8 neurons with sigmoid activation
- Output layer: 3 neurons with softmax activation (3 Iris species)

## Results
- Training accuracy: ~98%
- Test accuracy: ~96%


