import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import MNISTDataset

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

data = MNISTDataset(train_images.reshape([-1, 784]), train_labels,
                    test_images.reshape([-1, 784]), test_labels, batch_size=128)

train_steps = 500

learning_rate = 0.1


W1 = tf.Variable(tf.random.uniform([784, 512], minval=-0.1, maxval=0.1))
b1 = tf.Variable(tf.random.uniform([512], minval=-0.1, maxval=0.1))

W2 = tf.Variable(tf.random.uniform([512, 254], minval=-0.1, maxval=0.1))
b2 = tf.Variable(tf.random.uniform([254], minval=-0.1, maxval=0.1))

W3 = tf.Variable(tf.random.uniform([254, 128], minval=-0.1, maxval=0.1))
b3 = tf.Variable(tf.random.uniform([128], minval=-0.1, maxval=0.1))

W4 = tf.Variable(tf.random.uniform([128, 10], minval=-0.1, maxval=0.1))
b4 = tf.Variable(tf.random.uniform([10], minval=-0.1, maxval=0.1))


acc_plot, loss_plot = [], []


@tf.function
def train(img_batch, lbl_batch, learning_rate):
    with tf.GradientTape() as tape:

        logits1 = tf.nn.relu((tf.matmul(img_batch, W1) + b1))
        logits2 = tf.nn.relu((tf.matmul(logits1, W2) + b2))
        logits3 = tf.nn.relu((tf.matmul(logits2, W3) + b3))
        logits = tf.nn.softmax((tf.matmul(logits3, W4) + b4))  # output from training

        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=lbl_batch))  # Loss

    grads = tape.gradient(xent, [W1, W2, W3, W4, b1, b2, b3, b4])

    W1.assign_sub(learning_rate * grads[0])
    b1.assign_sub(learning_rate * grads[4])

    W2.assign_sub(learning_rate * grads[1])
    b2.assign_sub(learning_rate * grads[5])

    W3.assign_sub(learning_rate * grads[2])
    b3.assign_sub(learning_rate * grads[6])

    W4.assign_sub(learning_rate * grads[3])
    b4.assign_sub(learning_rate * grads[7])

    return xent, logits


for epoch in range(train_steps):

    img_batch, lbl_batch = data.next_batch()

    loss, logits = train(img_batch, lbl_batch, learning_rate=0.1)

    (loss_plot.append(loss.numpy()))

    preds = tf.argmax(logits, axis=1, output_type=tf.int32)

    acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))

    acc_plot.append(acc)

    print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")


def loss_plotting(train_steps, loss_plot):

    plt.style.use('ggplot')
    plt.plot(range(train_steps), loss_plot, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss vs Epochs")
    plt.show()


def acc_plotting(train_steps, acc_plot):

    plt.style.use('ggplot')
    plt.plot(range(train_steps), acc_plot, 'b')
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.show()


loss_plotting(train_steps, loss_plot)
acc_plotting(train_steps, acc_plot)


@tf.function
def test(test_data, test_labels):
    logits1_1 = tf.nn.relu((tf.matmul(test_data, W1) + b1))
    logits2_2 = tf.nn.relu((tf.matmul(logits1_1, W2) + b2))
    logits3_3 = tf.nn.relu((tf.matmul(logits2_2, W3) + b3))
    logits_2 = tf.nn.softmax((tf.matmul(logits3_3, W4) + b4))  # output from training
    test_preds = tf.argmax(logits_2, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, test_labels), dtype=tf.float32))
    return acc


accuracy = test(data.test_data, data.test_labels)
print(f"Accuracy on the test data: {accuracy}")
