# For main.py 
---

(Explanation)




Dataset:
----------------------------------------------

We are using the MNISTDataset class from datasets.py to create datasets via the wrapper function and the batch size is 128.
We are also doing something interesting here. We are flattening the data. into 784 dimensional vector space leading 
to MNIST images becoming bunch of points in 784 dimensional vector space. 
Also notice, by flatteing the data we are loosing spatial information of about the 2D structure of the image. 
Since we are going to use MLP which expects input as 1D vector we are doing this.
We can use the 2D strucutre of the data later when we are going to use CNN.

Related to reshapping:
Our training image is a 28,28 photo and if we look at the shape of the training image we will see we have 60000 photo. 
So we want to convert all these photos to a 28\*28 = 784 1D vector.
The argument of the reshape is [-1,784] - what this line will do is it will take the train_images
and reshape all the images to a 784 1D vector. The reason we have -1 is, we do not know how many images 
are thee in the train_images, placing -1 means take as much row as you need and 
on each row we will have a 1D vector of 784 value. If we knew the number of train_images
we have then we can pass the number of images instea of -1 (actually we do know the number of images in the train_images
which is 60000. And if we add [60000, 784] (for train_images) as the input to the reshape, it will perform the same thing)


Weights and biases:
----------------------------------------------

The first input layer has shape of [784, 512] because: we have input of 784 pixel value and we want the number neuron in the first hidden layer h1 to have
512 neuron. Each neuron in the h1 will have a bias, that is why we have 512 biases for each of that neuron. The second hidden layer h2 has 254 neuron, and 
as epected we will have 254 biases for 254 neuron.  The third hidden layer h3 has 128 neuron. The last layer i.e. the output layer has 10 output neuron since we are
doing classification for MNIST dataset and the total number of class is 10.


Training:
----------------------------------------------

We are performing train_steps number epochs and at each epoch we take a batch of training sample using next_batch() function. 
```
logits1 = output from first layer
logits2 = output from second layer
logits3 = output from third layer
logits  = output from last layer
```
We are performing matrix multiplication with the weights and biases and as an activation function we are using relu except the last layer  i.e. the output layer, 
since we are performing classification we would like to have probability of each of the 10 classes available.


Testing:
--------------------------------------------
Passing the test images through the network but now the weights and biases are updated from training. The calculating the accuracy
based on the true label and predicted label. 


For execution: Please go to the file location where main.py along with datasets.py has been saved, and from the command line execute the below command (tested on Python 3.7.4).

```
python main.py
```


# For with_tffunction.py
---
Implemented out of curiosity the same solution using @tf.function decorator to speed things up and wrapped code in functions for ease of use. 


For execution: Please go to the file location where with_tffunction.py along with datasets.py has been saved, and from the command line execute the below command (tested on Python 3.7.4).

```
python with_tffunction.py
```

# For another_approach.py 
---

Implemented out of curiosity the same solution using Keras layers API while maintaining the low level implementaion which includes custom training loop using tf.GradientTape, initializing weights and biases manually and others.

For execution: Please go to the file location where with_tffunction.py along with datasets.py has been saved, and from the command line execute the below command (tested on Python 3.7.4).

```
python another_approach.py
```
