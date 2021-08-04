# Five files that has errors based on the choice of model architecture, hyperparameters etc and fixing them (through visualization via tensorboard and intuition)
---


In the Fail folder we have five python file that fails due to poor choice of model architecture, hyperparameters etc. The task is to find the erros through visualization using tensorboard
and propose a solution for them. The errors has been fixed in the Solution folder. In each file inside this folder I have added small comment where the changes has been made.

---


### Fail1

---


Error: 

  * Vanishing gradient: due to high learning rate which does not allow the network to converge but to diverge.
  * The total number of neurons in each layer is extremely low, therefore mdoel is not being able to learn (this applies to all the fails in this Task! WHY ONLY 100 NEURON??!?!?!?!)
  * No need for such a deep network for a simple dataset, or else we will overfit the whole training samples.

Solution:

  * Decrease learning rate which allows convergence.
  * Increase total number of neuron leads to better learning.
  * Decrease n_layers = 3


### Fail2

---
Error: 
  * The activation function in the layers were sigmoid and sigmoid is not a good activation function since head and tail is almost 
  flat and leads to stale gradients which means there is no gradient in those region - leading to network parameters' update almost impossible, since there is no
  gradient the parameters are not updated. 

  Throuigh Visuzliation we can see the gradients of the weight metrices is extremely low! Therefore no learning 

Solution: 
  * Use relu which has linear activation for the positive part of the x axis. Leading to always having a gradient no matter what is the input (greater than zero) to the activation function.
  Through visualization we can see we will have much higher gradient now, leading to learning taking place!
  
### Fail3

---

Error:
* The weight initialization was not good, it was initialized between -0.01 and 0.0. Which is not a good choice since it doesnt allow the network to 
initizliae with weights that has more freedom to train. If we constrain the weights to 0 as lowest then the network is unable to learn complicated function
(since in the end a NN is a function approximator)
 
  Also as we know Relu has zero output for any negative input, leading to no gradient for those values - leading the network to
have weights that are initialized between -0.01 and 0.0 eventually leadinh to low gradient (we also have to take bias into account).

Solution: 
  * Good initialization of the weights is necessary instead of the given one, it should be -0.01 to +0.01


### Fail4

---

Error: 
  * Adding too much noise in during training that the image does not even exists - Through visualization we can observe that

Solution: 
  * Remove noise and the netowrk works perfectly


## Fail5

---

Error: 
* Does not make any sense to have output of softmax to be another input to a softmax to calculate the loss ( the loss is calculated in tf.GradientTape).

  It is like we are applying two softmax one after another. Because when we comptue the loss we are already applying softmax the output -

  according to tf documentation 

  ```
  tf.nn.sparse_softmax_cross_entropy_with_logits

  Warning: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. 
  Do not call this op with the output of softmax, as it will produce incorrect result
  ```

Solution: 
* Do not use a softmax in the last layer.
