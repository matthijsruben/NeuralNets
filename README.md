# NeuralNets
This repo contains code for several different neural networks. This project is created by Matthijs Kok and is used for a bachelor thesis of the study Business and IT on the University of Twente (academic purposes). In order to use/run this code, make sure you adhere to the [requirements](#requirements). For an explanation you can read the descriptions of the files below. For help, questions or remarks, please send an email to <email>. 

# Requirements:

There are a few things required, before you will be able to run the code published on this repository. a 64-bit Python version 3.5-3.7 is required, because of compatibility reasons with Tensorflow (library). The code was written in Python 3.6.2.
Next, the following libraries need to be installed and working: tensorflow, keras, numpy, matplotlib.

---

# Explanation of the files:


# ANNFashionMNIST.py

The file ANNFashionMNIST.py contains code for a neural network that can be trained on the dataset of fashionMNIST, which contains pictures of different types of clothes.
Short explanation of the code:
First, The data is divided into training data and test (validation) data. Since the data contains 28x28 pixel grayscale images (0-255), their values are first divided by 255, so that the grayscale value will be represented as a number between 0 (white) and 1 (black).
Then, the model of the Neural Network is created. There are 3 layers. The first layer is used for dimensionality reduction, such that the image is converted to a 784-dimensional vector. Each component of this vector will be used as input unit within the layer. The second layer contains 128 units which are activated using the 'relu' activation function. The third layer contains 10 units corresponding to the amount of classes (10 different cloths), which are activated using the 'softmax' activation function in order to acquire a probability used by the network to predict which type of cloth it think it is.
Next, the model is compiled using the optimizer 'adam', because 'adam' nicely incorporates both 'SGD with momentum' and 'RMSProp'. 'Momentum' uses the exponentially weighted moving average of a time series of gradients in order to accelerate the approach to the (local) minimum of the loss function and to dampen the oscillating effect in pathological curvatures on the surface of the loss function, such as ravines. 'RMSProp' automatically performs simulated annealing in the learning parameter, such that weight update steps become smaller when approaching a minimum. 'Adam' nicely combines the two, which is the reason it is chosen. Currently, 'sparse_categorical_crossentropy' is chosen as the loss function, which is a rather arbitrary choice. Also, the accuracy metric, which calculates how often prediction equals lables, is specified in order to keep track of its value each epoch.
Lastly, the trainig is started, where 5 epochs are specified (which is a rather low amount of epochs).
Finally, the first 5 images from the test (validation) data are visualised as an image, where both the label and prediciton of the network are displayed.

# ANNMNIST.py

The file ANNMNIST.py contains code for a neural network that can be trained on the dataset MNIST, which contains pictures of handwritten digits.
Short explanation of the code:
First, The data is divided into training data and test (validation) data. Since the data contains 28x28 pixel grayscale images (0-255), their values are first divided by 255, so that the grayscale value will be represented as a number between 0 (white) and 1 (black).
Then, the model of the Neural Network is created. There are 3 layers. The first layer is used for dimensionality reduction, such that the image is converted to a 784-dimensional vector. Each component of this vector will be used as input unit within the layer. The second layer contains 128 units which are activated using the 'relu' activation function. The third is a dropout layer with a parameter of 0.2. The last layer is a dense layer with 10 units.
