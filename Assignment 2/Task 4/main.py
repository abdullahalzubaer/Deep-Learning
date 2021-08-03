import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

'''
Playing around with shuffle, repeat and batch.
'''

features = tf.constant([[1, 3], [2, 1], [3, 3], [4, 4]])  # ==> 4x2 tensor
labels = tf.constant(['A', 'B', 'D', 'E'])  # ==> 4x1 tensor


train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))

train_dataset = train_dataset.repeat(3).shuffle(30).batch(4)
'''
Repeat: If less data do repeat
Shuffle: Keeps the original dataset but shuffles the data.
Batch: Takes batches of elements from the dataset sequentually therefore do shuffle then do batches

And if you need everything i.e. shortage of data and want want to shuffle and do batches then
Repeat first then do shuffle then batches because repeat will create copy of the original dataset and
make a larger dataset.
'''
for feature, labels in train_dataset:
    print(f"\nnext batch")
    print(f"Feature{feature}")
    print(f"Labels{labels}")

# MNIST part 1

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))


for t_image, t_label in train_data:
    # print(t_image.numpy())
    plt.imshow(t_image.numpy())
    print(t_label.numpy())
    break

# MNIST part 2

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_data = train_data.shuffle(500).batch(32)

test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
# now we have batches of data
for t_image, t_label in train_data:
    print(t_image.shape)
    print(t_label.shape)
    break

# MNIST part 3

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = (train_images.astype(np.float32)/255)
train_images.reshape((-1, 784))
train_labels = (train_labels.astype(np.int32))

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(256).batch(256)

for t_image, t_label in train_dataset:
    print(t_image.shape)
    print(t_label.shape)
    break
