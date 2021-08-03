import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Model(tf.Module):  # Inhereting tf module class and its functionality if required
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Init method from tf.Module
        self.W = tf.Variable(0.0)  # initializing the weight
        self.b = tf.Variable(0.0)  # initializing the bias

    def __call__(self, x):
        return self.W * x + self.b


def loss(target_y, predicted_y):  # We will minimize the loss
    return tf.reduce_mean(tf.square(target_y - predicted_y))


@tf.function  # with tf.funciton it takes 5s and without tf.function it takes 9 second -> for training with 5000 epochs--wohoo
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(outputs, model(inputs))

    dw, db = tape.gradient(current_loss, [model.W, model.b])

    model.W.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)


model = Model()

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs*TRUE_W + TRUE_b + noise


# Let'st rain the model for certain epochs and output the loss
"""
What we are doing here is we are keeping track of the weight bias and the loss in a list
and then we are appending the new weight, bias and the loss in the list and
then later after the for loop we are using it to make the plot.

"""

epochs = range(50)
Ws, bs, c_loss = [], [], []
print(f"Initial weight before training: {model.W.numpy()}")  # un-trained weight
print(f"Initial bias before training: {model.b.numpy()}")  # un-trained bias

for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(outputs, model(inputs))
    c_loss.append(current_loss)

    train(model, inputs, outputs, learning_rate=0.1)


print(f"Weight after training: {model.W.numpy():.5f}")  # trained weight
print(f"Bias after training: {model.b.numpy():.5f}")  # trained bias

plt.style.use('ggplot')
plt.plot(epochs, Ws, 'tab:red', label='Weight')
plt.plot(epochs, bs, 'tab:blue', label='Bias')
plt.xlabel('Epochs')
plt.ylabel('Weight and Bias')
plt.legend()
plt.show()

print("\n")

plt.plot(epochs, c_loss, 'tab:red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
