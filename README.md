**Image Classification with Convolutional Neural Networks (CNN)**
(All of this is subject to change as the scope of the project is fully flushed out)

**Project Overview**
This project demonstrates the application of Convolutional Neural Networks (CNN) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal of this project is to build and train a CNN model to accurately classify images into their respective categories.

**Prerequisites**
Python 3.6 or above
TensorFlow 2.x
Keras
NumPy
Matplotlib

**The CNN model used in this project consists of the following layers:**
(Could change as design is further developed)

Convolutional Layer with 32 filters and a kernel size of 3x3
Activation Layer (ReLU)
Max Pooling Layer with a pool size of 2x2
Convolutional Layer with 64 filters and a kernel size of 3x3
Activation Layer (ReLU)
Max Pooling Layer with a pool size of 2x2
Flatten Layer
Dense Layer with 64 units
Activation Layer (ReLU)
Dense Layer with 10 units (output layer)
Activation Layer (Softmax)

**Training**
The model is trained using the following parameters:

Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 10
Batch Size: 32

**Evaluation**
The model's performance is evaluated on the test set using accuracy as the metric. A confusion matrix is also plotted to visualize the model's performance across different classes.


**Future Work**
Experiment with different CNN architectures and hyperparameters to improve model performance.
Implement data augmentation techniques to increase the diversity of the training dataset and improve generalization.
Explore transfer learning by using pre-trained models to further enhance the model's accuracy.
