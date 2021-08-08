import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing import image
import numpy as np


# Getting the data.

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape([-1, 28, 28, 1]).astype(np.float32)/255
x_test = x_test.reshape([-1, 28, 28, 1]).astype(np.float32)/255

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_test_non_batch = x_test
y_test_non_batch = y_test


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
# I dont want batch less than 128 for test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128, drop_remainder=True)
print(len(x_test))


# Model 1 - Uncomment to use this model

# Create the NN model using Functional API for CNN
# inputs = keras.Input(shape=(28, 28, 1), name="input")
# x1 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), activation='relu')(inputs)
# x2 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), activation='relu')(x1)
# x3 = tf.keras.layers.Flatten()(x2)
# out = tf.keras.layers.Dense(10)(x3)
# model = tf.keras.Model(inputs=inputs, outputs=out)


# Model 2 

'''
Model 2 Result:

    Epoch = 2

    Training accuracy: 0.963
    Training loss    : 0.001

    Test accuracy    : 0.971
    Test loss        : 1.488
'''
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])


# Model 1 - Uncomment to use this model, it is an MLP (if using CPU maybe this is better)
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     # tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10)  # reason why no softmax previously written
# ])


optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# With @tf.function Start

@tf.function  # a lot faster
def train_step(input_x, input_y):
    with tf.GradientTape() as tape:
        logits = model(input_x)
        loss_val = loss_fn(input_y, logits)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_val  # if you need

# With @tf.function End


epochs = 2

for epoch in range(epochs):
    print(f"Epoch number: {epoch+1}")
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_val = train_step(x_batch_train, y_batch_train)
        if step % 10 == 0:
            prediction = probability_model(x_batch_train)
            test_acc = test_accuracy(y_batch_train, prediction).numpy()
            print(f"Step: {step}, Accuracy: {test_acc}, Loss: {loss_val}")


# Uncomment below for without tf.function
#without tf.function start#
# for epoch in range(epochs):
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#         with tf.GradientTape() as tape:
#             logits = model(x_batch_train)
#             loss_val = loss_fn(y_batch_train, logits)
#         grads = tape.gradient(loss_val, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         if step % 10 == 0:
#             prediction = probability_model(x_batch_train)
#             test_acc = test_accuracy(y_batch_train, prediction).numpy()
#             print(f"Step: {step}, Accuracy: {test_acc}, Loss: {loss_val}")
#without tf.function End#

# If you want to see how the model looks like
# keras.utils.plot_model(model, "mnist_model_cnn.png", show_shapes=True, expand_nested=True)
# model.summary()


###-----------Test-------------##

# calculating model's accuracy and loss with test data in two different ways
# 1. Whole test datasets
# 2. In batches


# 1. Whole test datasets
prediction = probability_model(x_test_non_batch)
loss_val = loss_fn(y_test_non_batch, prediction)
test_acc = test_accuracy(y_test_non_batch, prediction).numpy()

print(f"Test Samples (full dataset) -> Total Loss: {loss_val:.3f}, Accuracy: {test_acc:.3f}")


# # 2. In batches -> Uncomment below for calculating error in batches (in the end we are taking mean of all the batches loss and accuracy)
# mean_loss = list()
# mean_accuracy = list()
#
#
# for x_test, y_test in test_dataset:
#     prediction = probability_model(x_test)
#     loss_val = loss_fn(y_test, prediction)
#
#     mean_loss.append(loss_val.numpy())
#     mean_accuracy.append(test_accuracy(y_test, prediction).numpy())
#
# mean_loss = np.mean(mean_loss)
# mean_accuracy = np.mean(mean_accuracy)
#
# print(f"Test Samples (in batches) -> Total Loss: {mean_loss:.3f}, Accuracy: {mean_accuracy:.3f}")


###-----------Predicting a single image with the trained model-------------##

# To predict one single image from
# 1. disk or
# 2. from test images

# 1. From disk -> Uncomment for this
# # target size = resize the image as we are reading it
# img = image.load_img("/content/Ds5Rc.png", color_mode='grayscale',
#                      target_size=(28, 28, 1))  # color mode convert 3 to 1 channel
# img_array = keras.preprocessing.image.img_to_array(img)
# print(img_array.shape)
# img_array = img_array/255.
# # adding that extra dimention (batch) that the network expects
# img_array = tf.expand_dims(img_array, 0)
# print(img_array.shape)
# prediction_internet_image = probability_model.predict(img_array)
# print(f"Predicted image is {np.argmax(prediction_internet_image)}")


# 2. from test images
# to test an image from the test datset
img = x_test[4]  # selecting an image
# this is how the neural network expects input, with the first 1 means batch of 1.
img = (np.expand_dims(img, 0))
prediction_single = probability_model(img)
print(f"Predicted label: {np.argmax(prediction_single)}")
print(f"True label: {y_test[4]}")
