import numpy as np
import tensorflow as tf
import os
import time
import datetime


class MNISTDataset:
    """'Bare minimum' class to wrap MNIST numpy arrays into a dataset."""

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
            print("Starting new epoch...")
        return batch

    def shuffle_train(self):
        shuffled_inds = np.arange(self.train_data.shape[0])
        np.random.shuffle(shuffled_inds)
        self.train_data = self.train_data[shuffled_inds]
        self.train_labels = self.train_labels[shuffled_inds]


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
data = MNISTDataset(train_images.reshape([-1, 784]), train_labels,  # 28*28 = 784
                    test_images.reshape([-1, 784]), test_labels, batch_size=128)  # initializing MNISTDaataset class object (data)


W1 = tf.Variable(tf.random.uniform([784, 512], minval=-0.1, maxval=0.1))
b1 = tf.Variable(tf.random.uniform([512], minval=-0.1, maxval=0.1))

W2 = tf.Variable(tf.random.uniform([512, 254], minval=-0.1, maxval=0.1))
b2 = tf.Variable(tf.random.uniform([254], minval=-0.1, maxval=0.1))

W3 = tf.Variable(tf.random.uniform([254, 128], minval=-0.1, maxval=0.1))
b3 = tf.Variable(tf.random.uniform([128], minval=-0.1, maxval=0.1))

W4 = tf.Variable(tf.random.uniform([128, 10], minval=-0.1, maxval=0.1))
b4 = tf.Variable(tf.random.uniform([10], minval=-0.1, maxval=0.1))


current_dir = r"PATH_TO_LOG_DIRECTORY"  # r for raw string

logdir = os.path.join(current_dir + str(time.time()))  # Providing unique name to each log directory
train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
test_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))


def logging():
    '''
    A simple logging function for this Assignment.
    '''
    # Training:
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc_train = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))

    # Testing: Also testing with the test data as training progresses
    logits1_1 = tf.nn.relu((tf.matmul(data.test_data, W1) + b1))
    logits2_2 = tf.nn.relu((tf.matmul(logits1_1, W2) + b2))
    logits3_3 = tf.nn.relu((tf.matmul(logits2_2, W3) + b3))
    logits_2 = tf.nn.softmax((tf.matmul(logits3_3, W4) + b4))  # output from training

    test_preds = tf.argmax(logits_2, axis=1, output_type=tf.int32)
    acc_test = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels), dtype=tf.float32))

    with train_writer.as_default():

        # 5 image output  from training batch
        tf.summary.image("input", tf.reshape(img_batch, [-1, 28, 28, 1]), step=steps, max_outputs=5)

        tf.summary.scalar("accuracy_train", acc_train, step=steps)
        tf.summary.scalar('loss', xent, step=steps)

        # visuzliaing the weights how it changes in every 100th iteration
        tf.summary.histogram("weight_1", W2, step=steps)
        tf.summary.histogram("weight_2", W2, step=steps)
        tf.summary.histogram("weight_3", W3, step=steps)
        tf.summary.histogram("weight_4", W4, step=steps)

        # visuzliaing the gradient that we have for the weights only
        tf.summary.histogram("gradient_1_w", grads[0], step=steps)
        tf.summary.histogram("gradient_2_w", grads[1], step=steps)
        tf.summary.histogram("gradient_3_w", grads[2], step=steps)
        tf.summary.histogram("gradient_4_w", grads[3], step=steps)

        # just looking at the images nothing much that we are training the network with
        tf.summary.image("input", tf.reshape(img_batch, [-1, 28, 28, 1]), step=steps)

    with test_writer.as_default():

        tf.summary.scalar("accuarcy_test", acc_test, step=steps)


train_steps = 1000

learning_rate = 0.1

for steps in range(train_steps):
    img_batch, lbl_batch = data.next_batch()

    with tf.GradientTape() as tape:

        logits1 = tf.nn.relu((tf.matmul(img_batch, W1) + b1))
        logits2 = tf.nn.relu((tf.matmul(logits1, W2) + b2))
        logits3 = tf.nn.relu((tf.matmul(logits2, W3) + b3))
        logits = tf.nn.softmax((tf.matmul(logits3, W4) + b4))  # output from training

        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=lbl_batch))  # Average loss for a batch of sample

    # Gradient of all the weights anb biases w.r.t to loss
    grads = tape.gradient(xent, [W1, W2, W3, W4, b1, b2, b3, b4])

    # Updating the weights and biases w.r.t to the loss and learning rate
    W1.assign_sub(learning_rate * grads[0])
    b1.assign_sub(learning_rate * grads[4])

    W2.assign_sub(learning_rate * grads[1])
    b2.assign_sub(learning_rate * grads[5])

    W3.assign_sub(learning_rate * grads[2])
    b3.assign_sub(learning_rate * grads[6])

    W4.assign_sub(learning_rate * grads[3])
    b4.assign_sub(learning_rate * grads[7])

    # for logging in tensorboard in Colab

    if not steps % 100:

        # logging at every 100 steps:

        logging()
