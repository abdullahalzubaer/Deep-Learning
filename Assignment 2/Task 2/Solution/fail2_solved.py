import numpy as np
import tensorflow as tf
import os
import time
from datasets import MNISTDataset

current_dir = r"PATH_TO_LOG_DIRECTORY"  # r for raw string

logdir = os.path.join(current_dir + str(time.time()))  # Providing unique name to each log directory
train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
# test_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))

(train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.mnist.load_data()

mnist = MNISTDataset(train_imgs.reshape((-1, 784)), train_lbls,
                     test_imgs.reshape((-1, 784)), test_lbls,
                     batch_size=256)


# this is a super deep model, cool!
n_units = 100
n_layers = 8
w_range = 0.1

# just set up a "chain" of hidden layers
layers = []
for layer in range(n_layers):
    layers.append(tf.keras.layers.Dense(
        n_units, activation=tf.nn.relu,  # Changed from sigmoid activation to relu
        kernel_initializer=tf.initializers.RandomUniform(minval=-w_range,
                                                         maxval=w_range),
        bias_initializer=tf.initializers.constant(0.001)))

# finally add the output layer
layers.append(tf.keras.layers.Dense(
    10, kernel_initializer=tf.initializers.RandomUniform(minval=-w_range,
                                                         maxval=w_range)))

lr = 0.1
for step in range(2000):
    img_batch, lbl_batch = mnist.next_batch()
    with tf.GradientTape() as tape:
        # here we just run all the layers in sequence via a for-loop
        out = img_batch
        for layer in layers:
            out = layer(out)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=out, labels=lbl_batch))

    weights = [var for l in layers for var in l.trainable_variables]
    grads = tape.gradient(xent, weights)
    for grad, var in zip(grads, weights):
        var.assign_sub(lr*grad)

    if step % 100 == 0:

        preds = tf.argmax(out, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))
        print("Loss: {} Accuracy: {}".format(xent, acc))

        # Visualization start

        with train_writer.as_default():

            #-------------------------for input image----------------------------------#
            tf.summary.image("input_image", tf.reshape(
                img_batch, [-1, 28, 28, 1]), step=step, max_outputs=5)

            #-------------------------for accuracy and loss----------------------------------#
            tf.summary.scalar("accuracy", acc, step=step)
            tf.summary.scalar("loss", xent, step=step)

            #-------------------------for gradients----------------------------------#

            kernel_position_list = list()
            for weight in range(len(weights)):
                if "kernel" in weights[weight].name:
                    kernel_position_list.append(weight)

            for counter, kernel in enumerate(kernel_position_list):
                tf.summary.scalar(f"gradinet_of_weights_from_layer_{counter}", tf.norm(
                    grads[kernel]), step=step)

        # END


out = mnist.test_data
for layer in layers:
    out = layer(out)
test_preds = tf.argmax(out, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, mnist.test_labels), tf.float32))
print("Final test accuracy: {}".format(acc))
