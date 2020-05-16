import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Training data from fashionMNIST
data = keras.datasets.fashion_mnist
# Split training and validation data
(train_images, train_labels), (test_images, test_labels) = data.load_data()
# change to decimal representation
train_images = train_images/255.0
test_images = test_images/255.0

# Create a new Neural Network (NN)
model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),      # sigmoid, relu, softmax
    keras.layers.Dense(10, activation='softmax')
])

# Training specification (optimizer=SGD, loss-function=MSE, metrics=Accuracy)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Start Training
model.fit(train_images, train_labels, epochs=5)

# Evaluate performance
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# Models prediction
prediction = model.predict(test_images)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# show for 5 images in the test_images: the image, the label, the prediction
for i in range(5):
    plt.grid(False)  # turn off grid
    plt.imshow(test_images[i], cmap=plt.cm.binary)  # put binary image on plot
    plt.xlabel("Actual: " + class_names[test_labels[i]])  # actual label
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])  # predicted label
    plt.show()  # show the image


