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


# define the model first, from input to output

# this is a super deep model, cool!
n_units = 1000  # CHANGE HERE (from 100 to 1000)
n_layers = 3  # CHANGE HERE (from 8 to 3)
w_range = 0.4  # NOTE: lowering the w_range makes it even worse since less variation of weights available


# just set up a "chain" of hidden layers
layers = []
for layer in range(n_layers):
    layers.append(tf.keras.layers.Dense(
        n_units, activation=tf.nn.relu, kernel_initializer=tf.initializers.RandomUniform(
            minval=-w_range, maxval=w_range),
        bias_initializer=tf.initializers.constant(0.001)))


"""
output layer:

We do not need an activation in the last layer since we
are going to calculate the loss by using softmax function in the xent
that is why there is no activation function.
"""
layers.append(tf.keras.layers.Dense(
    10, kernel_initializer=tf.initializers.RandomUniform(minval=-w_range,
                                                         maxval=w_range)))

lr = 0.01  # Learning rate plays a huge role (from 0.1 to 0.01)
for step in range(1000):
    img_batch, lbl_batch = mnist.next_batch()
    with tf.GradientTape() as tape:
        out = img_batch
        for layer in layers:
            out = layer(out)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=out, labels=lbl_batch))

    # weights means all trainable parameters in this network.
    weights = [var for l in layers for var in l.trainable_variables]
    grads = tape.gradient(xent, weights)

    for grad, var in zip(grads, weights):
        var.assign_sub(lr*grad)

    # Provide the name of the trainable parameters (weights and biases in this case)
    # for k in weights:
    #     print(k.name)
    # break

    if step % 100 == 0:  # Every 100th step only

        preds = tf.argmax(out, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))
        print("Loss: {} Accuracy: {}".format(xent, acc))

        # Visualization start

        with train_writer.as_default():  # for training samples only

            #-------------------------for input image----------------------------------#
            tf.summary.image("input_image", tf.reshape(
                img_batch, [-1, 28, 28, 1]), step=step, max_outputs=5)

            #-------------------------for accuracy and loss----------------------------------#
            tf.summary.scalar("accuracy", acc, step=step)
            tf.summary.scalar("loss", xent, step=step)

            #-------------------------for gradients----------------------------------#

            # # Gradient of all trainable parameter - Biases and Weights(another name: kernel) in this network we have 4 weights and 4 biases
            # for count, gradient in enumerate(grads):
            #     # tf.norm(gradient) gives overall impression of the gradient
            #     tf.summary.scalar(f"Grads_{count}", tf.norm(gradient), step=step)

            #---Gradient for only weights---#

            # look into the name of the trainable variables and look for where it has "kernel" - because
            # it means that is a weight. Kernel is another name of weight.

            """
            Loop through all trainable parameters in this case it is 8.

            If the trainable parameter has "kernel" string in it then add the position of that weight kernel that we have
            from the trainable_parameter's list into another list.

            We are just keeping track of the position in the trainable parameter's where it is weight.
            """

            kernel_position_list = list()
            for weight in range(len(weights)):
                if "kernel" in weights[weight].name:
                    kernel_position_list.append(weight)

            for counter, kernel in enumerate(kernel_position_list):
                tf.summary.scalar(f"gradinet_of_weights_from_layer_{counter}", tf.norm(
                    grads[kernel]), step=step)

            #---Gradient for only biases---#

            # look into the name of the trainable variables and look for where it has "bias" - because
            # it means that is a bias.

            """
            Loop through all trainable parameters in this case it is 8.

            If the trainable parameter has "bias" string in it then add the position of that weight kernel that we have
            from the trainable_parameter's list into another list.

            We are just keeping track of the position in the trainable parameter's where it is biases.
            """

            bias_position_list = list()
            for weight in range(len(weights)):
                if "bias" in weights[weight].name:
                    bias_position_list.append(weight)

            for counter, bias in enumerate(bias_position_list):
                tf.summary.scalar(
                    f"gradinet_of_bias_from_layer_{counter}", tf.norm(grads[bias]), step=step)

            #-------------------------for weights/biases----------------------------------#

            for count, i in enumerate(range(len(layers))):
                tf.summary.histogram(f"weights{count}", (layers[i].get_weights()[
                                     0]), step=step)  # [0]=weights, [1]=bias


out = mnist.test_data
for layer in layers:
    out = layer(out)
test_preds = tf.argmax(out, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, mnist.test_labels), tf.float32))
print("Final test accuracy: {}".format(acc))
