import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import (
    AvgPool2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAvgPool2D,
    Input,
    MaxPool2D,
    ReLU,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy

cifar100 = tf.keras.datasets.cifar100

(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)

# Composite Function
# x = tensor
# filter = number of filter for the cnn layer
# kernel_size = kernel size of the cnn layer


def composite_function(x, filters, kernel_size):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    return x


# DenseBlock
# k = number of filter for conv operation


def dense_block(tensor, k, reps):
    for _ in range(reps):
        x = composite_function(
            tensor, 4 * k, 1
        )  # BottleNeck. 4*k = filter, 1 = kernel size 1X1 filter
        x = composite_function(x, k, 3)  # 3X3 filter
        tensor = Concatenate()([tensor, x])
    return tensor


# Trnsition Layer
# f = number of filter
# theta = Compression factor


def transition_layer(x, theta):
    f = int(tf.keras.backend.int_shape(x)[-1] * theta)
    x = composite_function(x, f, 1)  # 1X1 cnn
    x = AvgPool2D(2, strides=2, padding="same")(x)  # 2 = pool size,
    return x


# Creating the model

k = 32  # Number of filter
theta = 0.5  # Compression
repetitions = 6, 12, 24, 16  # Total number of "conv" in respective blocks

# Input layer
input = Input(shape=(32, 32, 3))
x = Conv2D(2 * k, 7, strides=2, padding="same")(input)
x = MaxPool2D(3, strides=2, padding="same")(x)

# Main model (dense and transition)

for reps in repetitions:
    d = dense_block(x, k, reps)
    x = transition_layer(d, theta)


# Classification layer

x = GlobalAvgPool2D()(d)
output = Dense(100, activation="softmax")(x)


# creating the model

model = Model(input, output)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=["accuracy"],
)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=128,
)


# Evaluating on the test data
results = model.evaluate(test_images, test_labels, batch_size=128)
print(f"Test loss: {results[0]}, Test accuracy: {results[1]}")


'''
Model Evaluation:
     
    Platform: Colaboratory using GPU.

    Epoch = 10

    Training accuracy: 0.7513
    Training loss    : 0.8085 
    
    Test accuracy    : 0.4152
    Test loss        : 2.8083
'''
