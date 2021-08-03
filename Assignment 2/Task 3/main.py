# Using tf.data and custom Training loop and layers.

import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Extremely important to scale the images between 0 and 1.
x_train = (x_train.astype(np.float32)/255.).reshape((-1, 784))
y_train = y_train.astype(np.int32)  # Tf expects label as int32 or int64

x_test = (x_test.astype(np.float32)/255.).reshape((-1, 784))
y_test = y_test.astype(np.int32)


# Creating x and y from the dataset using from_tensor_slices. Each x_train has its corresponding y_train.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(256)

n_units = 1000  # Number of units for each layer except the last layer
n_layers = 3
w_range = 0.4
lr = 0.01
layers = list()


# This will create layer with what it is initialized with
for layer in range(n_layers):

    layers.append(tf.keras.layers.Dense(
        n_units, activation=tf.nn.relu, kernel_initializer=tf.initializers.RandomUniform(
            minval=-w_range, maxval=w_range),
        bias_initializer=tf.initializers.constant(0.001)))


layers.append(tf.keras.layers.Dense(10, kernel_initializer=tf.initializers.RandomUniform(minval=-w_range,
                                                                                         maxval=w_range)))

epoch = 2
for epochs in range(epoch):
    for img_batch, lbl_batch in train_dataset:

        with tf.GradientTape() as tape:
            out = img_batch
            # here we just run all the layers in sequence via a for-loop
            for layer in layers:
                out = layer(out)

            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=out, labels=lbl_batch))
            # get the trainable parameter out of the layers and store in a list
            weights = list()
            for l in layers:
                for var in l.trainable_variables:
                    weights.append(var)
            # calculating gradient all the trainable parameter w.r.t loss
            grads = tape.gradient(xent, weights)
            for grad, var in zip(grads, weights):
                var.assign_sub(lr*grad)  # update the parameters

            preds = tf.argmax(out, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))

    print(f"Epoch {epochs} Accuracy: {acc.numpy()} Loss: {xent.numpy()} ")


# To test

out_test = x_test  # getting the test data

for layer in layers:
    out_test = layer(out_test)  # passing the test data

test_preds = tf.argmax(out_test, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, y_test), tf.float32))
print(f"Test Accuracy: {acc.numpy()}")
