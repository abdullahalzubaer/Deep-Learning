# A simple linear regression model implemented at a very low level (custom training loop with tf.GradientTape()) using Tensorflow 2.0.

It is mostly for my understanding (with extensive comments regarding all the details, explained clearly: here) and how things work at a very low level.



Task
----------------------------------------------
You'll create a simple linear model, f(x) = x * W + b, which has two variables: W (weights) and b (bias).
You'll synthesize data such that a well trained model would have W = 3.0 and b = 2.0


tf.Variable vs tf.Constant
----------------------------------------------
Why we are using tf.Variable and not tf.Constant. The reason the value of a tensor is immutable except tf.Variable.
Since we want to update the weight and bias we have to use tf.Variable, remember this small tip bro.


Loss Function
----------------------------------------------
tf.square: This takes a tensor which is of 1D and returns the square of each element in the tensors.
tf.reduce_mean: This returns the mean (a single value) of the elements in the tensors.
tf.reduce_mean(tf.square(a_tensor):This is the mean square error of a_tensors, it is used in linearregression as a loss function


Dataset
----------------------------------------------

We have created inputs and noise from normal distribution.
The equation that we are trying to implement in this task is f(x) = x * W + b with W = 3.0, and b = 2.0,
we will try to learn this equation from some data. And to get the data we have created the synthetic data below.
We have also addded some noise.

We are going to use the synthetic data, we have to see what is the real output was supposed to be.
But the problem here is we are going to use some noise too.
And for that reason we will actually not have the perfect output even if we use the TRUE_W and TRUE_b.


Workflow
----------------------------------------------
We wre going to update the initialzied W and b of the model. For this we are going to use machine learning.

We will get the real output and the output by model. Then we are going to calculate the loss.
And then we are going to run an optimization method. We are going to use gradient descent optimization.

We will calculate the gradient of the loss w.r.t the parameter we have which is w and b.
And then we will use the gradient that we will have to change the value of the w and b.
And then with more and more iteration we will update the w and b.

By using gradient descent as the learning algorithm and by utilizing error backpropagation approach,
we eventually go to a place where the loss is minimum and then that is our w and b value.



Training method
----------------------------------------------

We are keeping track of the loss in current_loss. so for all the inputs to the model and the actual output
we will have loss and we are gonna store in the tape.
when we are outside the tf.GradientTape() we will have all the gradient that we will need to update the parameters.
And we are actually going to do this thing for several epochs.

We are storing the gradient wrt to the current loss for all the trainable parameters.
In our case the parameters are W and b.

We are updating the value of weight and bias by accessing the gradient and multiplying it with the learning rate,
which is passed to the fucntion assign_sub. And this function combines the assignment
and subtracting from the W and b according to the proportion of the gradient (opposite direction of where the gradient is pointing
since we are going to minmize the loss) and learning rate.

Also remember we are not using batches as input here, we are using the full dataset for training.

---

For execution: Please go to the file location where main.py has been saved, and from the command line execute the below command (tested on Python 3.7.4).

```
python main.py
```


---
Reference: https://www.tensorflow.org/guide/basic_training_loops
