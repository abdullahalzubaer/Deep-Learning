import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.reshape([-1, 32, 32, 3]).astype(np.float32) / 255
x_test = x_test.reshape([-1, 32, 32, 3]).astype(np.float32) / 255

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_test_non_batch = x_test
y_test_non_batch = y_test

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
    128, drop_remainder=True
)


optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(128, 2, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(64, 2, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(32, 2, padding="same", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


@tf.function
def train_step(input_x, input_y):
    with tf.GradientTape() as tape:
        logits = model(input_x)
        loss_val = loss_fn(input_y, logits)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_val


epochs = 10

for epoch in range(epochs):
    # print(f"Epoch number: {epoch+1}")

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_val = train_step(x_batch_train, y_batch_train)
        if step % 10 == 0:
            prediction = probability_model(x_batch_train)
            test_acc = test_accuracy(y_batch_train, prediction).numpy()

    print(f"Accuracy at end of epoch {epoch+1} and last batch: {test_acc:.3f}")

# Test: Uncomment for without tf.function : Test done in batches
"""
mean_loss = list()
mean_accuracy = list()

for x_test, y_test in test_dataset:
    prediction = probability_model(x_test)
    loss_val = loss_fn(y_test, prediction)

    mean_loss.append(loss_val.numpy())
    mean_accuracy.append(test_accuracy(y_test, prediction).numpy())


mean_loss = np.mean(mean_loss)
mean_accuracy = np.mean(mean_accuracy)

print(f"Test samples (in batches) -> Total Loss: {mean_loss:.3f}, Accuracy: {mean_accuracy:.3f}")
"""

"""
Test with tf.function:

Note: Cannot use numpy in tf.function (since it's a python thing)
"""


# @tf.function
# def test(x_test_non_batch):
#     prediction = probability_model(x_test_non_batch)
#     loss_val = loss_fn(y_test_non_batch, prediction)
#     test_acc = test_accuracy(y_test_non_batch, prediction).numpy()
#     return test_acc
# print(test(x_test_non_batch))


@tf.function
def test_step(x, y):

    test_accuracy.reset_state()
    prediction = probability_model(x)
    test_accuracy.update_state(y, prediction)
    acc = test_accuracy.result()
    loss = loss_fn(y, prediction)
    return (acc, loss)


for x_test, y_test in test_dataset:
    test_acc, test_loss = test_step(x_test, y_test)
    print(f"Test accuracy: {test_acc.numpy()} Test loss: {test_loss.numpy()}")


print(f"Final test accuracy: {test_acc:.4f} and loss: {test_loss:.3f}")
