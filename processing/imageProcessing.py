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

# Sequential is the way we tell Keras that we want a simple model where data flows linearly from one layer to the next
# Sequential is the blueprint for building a neural network
from keras.models import Sequential

# Conv2D: Image Extraction. MaxPooling2D: simplifies features. Flatten: Prepares data to feed into the NN. Dense: Classic NN layer for decision making
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# to_categorical allows us to turn our labels into numerical representations that can be better understood by our output layer
from keras.utils import to_categorical

# First we want to load all of the CIFAR images and divide them neatly so we have some to train and some to test
# We use tuples to handle the image representation and its associated label, this allows us to have labeled images to train and labels to check during testing
""" 
    Cifar10 "load_data" function returns 2 tuples:
    image_train - A numpy array containing the training images as a numerical representation based on pixel values from 0-255
    label_train - a numpy array containing the associated labels for the training images
    image_test - A numpy array containing the testing images as a numerical representation based on pixel values from 0-255
    label_test - a numpy array containing the associated labels for the testing images
    We have the same setup for our testing variables, the load_data function neatly divides the images and labels for us.
"""
(image_train, label_train), (image_test, label_test) = cifar10.load_data()

# One-hot encode the labels: This must happen after we load data but BEFORE we train the model
label_train = to_categorical(label_train)
label_test = to_categorical(label_test)

# Next, we need to convert the images pixel values into a numerical representation to be better processed by our model
# "astype('float32')" converts the pixel values from integers to floats for better calculations, we divide by 255 to scale the pixel values from 0-1 rather than 0-255
image_train = image_train.astype('float32') / 255.0
image_test = image_test.astype('float32') / 255.0

# We say sequential because we are building a linear stack of layers, image data flows frome one layer to the next
model = Sequential()


""" 
    The first layer is the feature extractor 
    it is a set of 32 image filters, each 3x3 pixels in size.
    Imagine it as a window that slides across the image to capture every 3x3 square of pixels
    This layer allows us to identify basic features like edges and curves
"""
# "model.add" is how we add layers to the model, setting up each layer for a specific functionality
# Relu is used to allow our model to detect more complex, non linear features
# Input shape allows us to give the dimensions of the image, with the color channels: 32 x 32 pixels with RGB color channels
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

"""
    Our next layer will do the "downsampling", this will reduce the size of the feature representations (from the first layer)
    from 3x3 representations to 2x2. Think of compression, where we reduce filesizes while keeping the important information
"""
model.add(MaxPooling2D(pool_size=2)) 

# Now we will do more feature recognition, but with 64 filters for even more powerful detection
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Once again, we will do the downsampling, keeping the important features of the feature recognition,
model.add(MaxPooling2D(pool_size=2)) 

# This next layer will take all the features learned by the Convolutions and spreads them into a single long vector, this will be important in the coming layers
model.add(Flatten())

# This is a fully connected layer, each node here connects to all nodes from the previous layer
# This does some high-level pattern mixing to try and make sense of the extracted features
model.add(Dense(64, activation='relu'))

# This is the final layer, we use 10 nodes to denote the 10 classes
model.add(Dense(10))

"""
    Optimizer: determines how aggressivley our model learns from our mistakes
    Loss: Allow the model to quantify how "wrong" it gets an image guess, this will affect the optimizer
    Metrics: Give us feedback during training on things like accuracy

    1.) Adam is a robust, but simple and forgiving optimizer thats great for self tuining
    2.) We want to quantify "how wrong" a response is, some wrong answers are better than others
    3.) We want to keep track of what % of images were correctly identified, we can tweak the model filters based on this metric
"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Now we will begin the training phase
"""
    We will get batches of images from "image_train" and their true labels from "label_train"
    It will predict, get the error (loss), and then use that to optimize its weights
    1.) Epochs: How many times do we go through the entire training set, usually many times
    2.) Batch_size: How many images the model sees before calculating a single weight update
    3.) Validation data: Allows the model to see how its doing with a "seperate" data set
"""
model.fit(image_train, label_train, epochs=10, batch_size=32, validation_data=(image_test, label_test))

# Next is the evaluation phase
"""
    Here, we get to see a performace report on data that the model NEVER saw during training
    test_loss: indicate how "wrong" the model's predictions tend to be
    test_acc: the accuracy metric, not the only metric that matters, but higher is better
"""
test_loss, test_acc = model.evaluate(image_test, label_test)
print('Model Accuracy: ', test_acc)

