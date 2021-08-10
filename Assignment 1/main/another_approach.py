# using MNISTDataset class and custom  training loop and layers.

import numpy as np
import tensorflow as tf
import os
import time
import datetime


class MNISTDataset:
    """'Bare minimum' class to wrap MNIST numpy arrays into a dataset."""
    # This block (MNISTDataset class) of code was provided to us in the assignment to work with the dataset without using tf.data module)

    def __init__(self, train_imgs, train_lbs, test_imgs, test_lbls, batch_size,
                 to01=True, shuffle=True, seed=None):
        """
        Use seed optionally to always get the same shuffling (-> reproducible
        results).
        """
        self.batch_size = batch_size
        self.train_data = train_imgs
        self.train_labels = train_lbs.astype(np.int32)
        self.test_data = test_imgs
        self.test_labels = test_lbls.astype(np.int32)

        if to01:
            # int in [0, 255] -> float in [0, 1]
            self.train_data = self.train_data.astype(np.float32) / 255
            self.test_data = self.test_data.astype(np.float32) / 255

        self.size = self.train_data.shape[0]

        if seed:
            np.random.seed(seed)
        if shuffle:
            self.shuffle_train()
        self.shuffle = shuffle
        self.current_pos = 0

    def next_batch(self):
        """Either gets the next batch, or optionally shuffles and starts a
        new epoch."""
        end_pos = self.current_pos + self.batch_size
        if end_pos < self.size:
            batch = (self.train_data[self.current_pos:end_pos],
                     self.train_labels[self.current_pos:end_pos])
            self.current_pos += self.batch_size
        else:
            # we return what's left (-> possibly smaller batch!) and prepare
            # the start of a new epoch
            batch = (self.train_data[self.current_pos:self.size],
                     self.train_labels[self.current_pos:self.size])
            if self.shuffle:
                self.shuffle_train()
            self.current_pos = 0
            # print("Starting new epoch...")
        return batch

    def shuffle_train(self):
        shuffled_inds = np.arange(self.train_data.shape[0])
        np.random.shuffle(shuffled_inds)
        self.train_data = self.train_data[shuffled_inds]
        self.train_labels = self.train_labels[shuffled_inds]


# get the data
(train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.mnist.load_data()
mnist = MNISTDataset(train_imgs.reshape((-1, 784)), train_lbls,
                     test_imgs.reshape((-1, 784)), test_lbls, batch_size=256)

n_units = 1000
n_layers = 3
w_range = 0.4
lr = 0.01
layers = list()

# Create layers

# This will create layer with what it is initialized with
for layer in range(n_layers):

    layers.append(tf.keras.layers.Dense(
        n_units, activation=tf.nn.relu, kernel_initializer=tf.initializers.RandomUniform(
            minval=-w_range, maxval=w_range),
        bias_initializer=tf.initializers.constant(0.001)))


layers.append(tf.keras.layers.Dense(10, kernel_initializer=tf.initializers.RandomUniform(minval=-w_range,
                                                                                         maxval=w_range)))

# Now we have n_layers with a last layer also.

# Training starts here
for steps in range(1000):

    img_batch, lbl_batch = mnist.next_batch()

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

        if steps % 100 == 0:
            print(f"Accuracy: {acc.numpy()} Loss: {xent.numpy()}")

out_test = mnist.test_data  # getting the test data

for layer in layers:
    out_test = layer(out_test)  # passing the test data

test_preds = tf.argmax(out_test, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, mnist.test_labels), tf.float32))
print(f"Test Accuracy: {acc.numpy()}")
