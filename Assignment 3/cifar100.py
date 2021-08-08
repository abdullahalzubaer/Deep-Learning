import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing import image
import numpy as np


# Getting the data.

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

x_train = x_train.reshape([-1, 32, 32, 3]).astype(np.float32)/255
x_test = x_test.reshape([-1, 32, 32, 3]).astype(np.float32)/255

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_test_non_batch = x_test
y_test_non_batch = y_test


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
# I dont want batch less than 128 for test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128, drop_remainder=True)


optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# very deep and complicated model
'''

Model Result:

    Epoch = 10

    Training accuracy: 0.755
    Training loss    : 0.080
    
    Test accuracy    : 0.711
    Test loss        : 4.370

'''
model = tf.keras.Sequential([
                            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
                            tf.keras.layers.Conv2D(128, 2, padding='same', activation='relu'),
                            tf.keras.layers.Conv2D(64, 2, padding='same', activation='relu'),
                            tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(1024, activation='relu'),
                            tf.keras.layers.Dropout(1.3),
                            tf.keras.layers.Dense(100)
                            ])


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# With @tf.function Start


@tf.function  # a lot faster
def train_step(input_x, input_y):
    with tf.GradientTape() as tape:
        logits = model(input_x)
        loss_val = loss_fn(input_y, logits)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_val  # if you need

# With @tf.function End


epochs = 10

for epoch in range(epochs):
    print(f"Epoch number: {epoch+1}")
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_val = train_step(x_batch_train, y_batch_train)
        if step % 10 == 0:
            prediction = probability_model(x_batch_train)
            test_acc = test_accuracy(y_batch_train, prediction).numpy()
            print(f"Step: {step}, Accuracy: {test_acc}, Loss: {loss_val}")


# # 2. In batches -> Uncomment below for calculating error in batches (in the end we are taking mean of all the batches loss and accuracy)
mean_loss = list()
mean_accuracy = list()


for x_test, y_test in test_dataset:
    prediction = probability_model(x_test)
    loss_val = loss_fn(y_test, prediction)

    mean_loss.append(loss_val.numpy())
    mean_accuracy.append(test_accuracy(y_test, prediction).numpy())

mean_loss = np.mean(mean_loss)
mean_accuracy = np.mean(mean_accuracy)

print(f"Test Samples (in batches) -> Total Loss: {mean_loss:.3f}, Accuracy: {mean_accuracy:.3f}")
