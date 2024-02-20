**Image Classification with Convolutional Neural Networks (CNN):**  

**Overview:**  
This project demonstrates the application of Convolutional Neural Networks (CNN) for image classification using the MNIST Dataset. The MNIST dataset is comprised of 70,000 images of handwritten numerical digets ranging from 0-10, at a size of 28x28 pixels with 1 color channel. The goals of this project are to get more comfortable writing complex python code with multiple libraries, introduce myself to ML concepts such as CNNs, and explore new areas of the field to prepare myself for more ML based projects.

**Tools Needed:** 
Python 3.6 or above  
TensorFlow 2.x  
Keras  
NumPy  
Matplotlib  

**The CNN model used in this project consists of the following layers:**  

Convolutional Layer with 32 filters and a kernel size of 3x3  
Activation Layer (ReLU)  
Max Pooling Layer with a pool size of 2x2  
Convolutional Layer with 64 filters and a kernel size of 3x3  
Convolutional Layer with 128 filters and a kernel size of 3x3  
Convolutional Layer with 256 filters and a kernel size of 3x3  
Convolutional Layer with 512 filters and a kernel size of 3x3  
Flatten Layer  
Dense Layer with 64 units  
Activation Layer (ReLU)  
Dense Layer with 10 units (output layer)  
Activation Layer (Softmax)  
Batch Normalization  
Dropout(0.2)  

**Training:**  
The model is trained using the following parameters:  

Optimizer: SDG (Learning rate =0.05)  
Loss Function: Categorical Crossentropy  
Metrics: Accuracy  
Epochs: 30  
Batch Size: 16  

**Evaluation:**  
The model's performance is evaluated on the test set using accuracy as the metric, with success being determied by >95% accuracy  

**Future Work:**  
Experiment with different CNN architectures and hyperparameters to improve model performance.  
Implement data augmentation techniques to increase the diversity of the training dataset and improve generalization.  
Explore transfer learning by using pre-trained models to further enhance the model's accuracy.  

**Why did I learn this?**  
I want to get a better understanding of more widely used programming languages like python which are great for ML due to the increase in demand for ML/AI skills.    
I want to use this project as a jumping off point to start to develop more complex image recognitnion apps, such as fashion apps which can use image recogniton to recomend outfits, this is the first step in that process.  

**Project Status:**    
The project was a success, after several rounds of tweaking, testing, and refinement, I was able to successfully reach 99% accuracy, I will continue to develop the model however to deal with more advanced data sets and images, and studying more terminology and techniques to make myself a stronger developer
