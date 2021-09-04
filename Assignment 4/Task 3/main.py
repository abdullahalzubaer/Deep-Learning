import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential(
    [
        InputLayer(input_shape=(32, 32, 3)),
        Conv2D(128, 2, padding="same", activation="relu"),
        Conv2D(64, 2, padding="same", activation="relu"),
        Conv2D(32, 2, padding="same", activation="relu"),
        Flatten(),
        Dense(1024, activation="relu"),
        Dense(10),
    ]
)

loss_fn = SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])


history = model.fit(x_train, y_train, validation_split=0.2, epochs=10)

result = model.evaluate(x_test, y_test, batch_size=128)
print(f"Test loss: {result[0]}, Test accuracy: {result[1]}")

plt.style.use("ggplot")
plt.plot(history.history["accuracy"], label="accuracy", color="tab:red")
plt.plot(history.history["val_accuracy"], label="val_accuracy", color="tab:blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
# plt.savefig("1.pdf")
plt.show()

plt.plot(history.history["loss"], label="loss", color="tab:red")
plt.plot(history.history["val_loss"], label="val_loss", color="tab:blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower right")
plt.show()
# plt.savefig("2.jpg")
