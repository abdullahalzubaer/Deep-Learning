import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import MNISTDataset

mnist = tf.keras.datasets.mnist

# getting as 2D matrix for img, 1D for lbl.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


data = MNISTDataset(train_images.reshape([-1, 784]), train_labels,  # 28*28 = 784
                    test_images.reshape([-1, 784]), test_labels, batch_size=128)  # initializing MNISTDaataset class object (data)

train_steps = 1000

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

w1_plot, w2_plot, w3_plot, w4_plot = [], [], [], []
b1_plot, b2_plot, b3_plot, b4_plot = [], [], [], []

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

    loss_plot.append(xent)

    # now for accuracy we have to predict first and then measure the accuracy

    preds = tf.argmax(logits, axis=1, output_type=tf.int32)

    acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))

    acc_plot.append(acc)
    print("Loss: {} Accuracy: {}".format(xent, acc))


plt.style.use('ggplot')
plt.plot(range(train_steps), loss_plot, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss vs Epochs")
plt.show()


plt.style.use('ggplot')
plt.plot(range(train_steps), acc_plot, 'b')
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.show()


# Now we are going to predict mnist digits using our trained model
logits1_1 = tf.nn.relu((tf.matmul(data.test_data, W1) + b1))
logits2_2 = tf.nn.relu((tf.matmul(logits1_1, W2) + b2))
logits3_3 = tf.nn.relu((tf.matmul(logits2_2, W3) + b3))
logits_2 = tf.nn.softmax((tf.matmul(logits3_3, W4) + b4))  # output from training

test_preds = tf.argmax(logits_2, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels), dtype=tf.float32))

print(f"Accuracy for the test dataset: {acc.numpy()}")
