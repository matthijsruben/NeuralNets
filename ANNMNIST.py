import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import tensorboard

# data
mnist = tf.keras.datasets.mnist
# split data into training and validation data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# train_data is an array that contains the different images (the training examples) as elements.
# Each element (image) is represented as an array that contains
# greyscale values of 0 (white) to 255 (black) on each entry.
# train_labels is an array that contains the correct target values (a number between 0 and 10) that labels the
# corresponding image in train_data.

# to make values of 0 to 1, instead of 0 to 255, I divide by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Training specification (optimizer=SGD, loss-function=MSE, metrics=Accuracy)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
              loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])

# Defining the Keras TensorBoard callback, for logging/graphing purposes.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Start Training
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])