# Executed this file in colab.

from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

!rm -rf ./logs/

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)




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


epochs = 1
tf.summary.trace_on(graph=True, profiler=True)
for epoch in range(epochs):
    # print(f"Epoch number: {epoch+1}")

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # tf.summary.trace_on(graph=True, profiler=True)
        loss_val = train_step(x_batch_train, y_batch_train)
        if step % 10 == 0:
            prediction = probability_model(x_batch_train)
            test_acc = test_accuracy(y_batch_train, prediction).numpy()

    print(f"Accuracy at end of epoch {epoch+1} and last batch: {test_acc:.3f}")

with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)

%load_ext tensorboard
%tensorboard --logdir logs/func
