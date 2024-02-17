"""
Nicholas Capriotti
2/17/2024
Tensorflow Image Recognition
This is a basic image recognition program to get introduced to tensorflow and machine learning using the CIFAR-10 Library
The goal is to correctly identify images with at least 95% accuracy
"""

# Import tensorflow for our deep learning toolkit
# "pip install" all necessary libraries
import tensorflow as tf

# The Tensorflow library "Keras" has the cifar10 dataset
# We want to import the dataset from the keras library, which was imported with tensorflow
# This dataset contains the thousands of images we will use to both train and test our model
from keras.datasets import cifar10

# Numpy is going to give us various mathematical tools and functions that are key for this type of high level processing
import numpy as np

# We use Matplotlab to let us vizualise things with plotting tools.
import matplotlib.pyplot as plt

# First we want to load all of the CIFAR images and divide them neatly so we have some to train and some to test
# We use tuples to handle the image representation and its associated label, this allows us to have labeled images to train and labels to check during testing
(image_train, label_train), (image_test, label_test) = cifar10.load_data()
""" 
    Cifar10 "load_data" function returns 2 tuples:
    image_train - A numpy array containing the training images as a numerical representation based on pixel values from 0-255
    label_train - a numpy array containing the associated labels for the training images
    image_test - A numpy array containing the testing images as a numerical representation based on pixel values from 0-255
    label_test - a numpy array containing the associated labels for the testing images
    We have the same setup for our testing variables, the load_data function neatly divides the images and labels for us.
"""

# Next, we need to convert the images pixel values into a numerical representation to be better processed by our model
# "astype('float32')" converts the pixel values from integers to floats for better calculations, we divide by 255 to scale the pixel values from 0-1 rather than 0-255
image_train = image_train.astype('float32') / 255.0
image_test = image_test.astype('float32') / 255.0

