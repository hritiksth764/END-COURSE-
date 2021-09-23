# ASSIGNMENT_01 # 
# 1. What is a neural network neuron?

**Neuron:**

-  It takes the **inputs** and multiplies them by their **weights,**
-  then it **sums** them up,
-  after that it applies the **activation function** to the sum.
-  A **neuron** in a neural network can be better understood with the help of biological neurons. An artificial neuron is similar to a biological neuron. It receives input from the other neurons, performs some processing, and produces an output.

![Screenshot-from-2021-02-25-13-43-51](https://user-images.githubusercontent.com/42655809/134331764-502107dd-4c88-44e4-9ce5-9a5a5ec17cfc.jpg)



Here, X1 and X2 are inputs to the artificial neurons, f(X) represents the processing done on the inputs and y represents the output of the neuron.


# 2. What is the use of the learning rate?

**Learning Rate:** 
The amount that the weights are updated during training is referred to as the step size or the **learning rate**.
Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0.
The learning rate controls how quickly the model is adapted to the problem. Smaller learning rates require more given the smaller changes made to the weights each update, whereas larger learning rates result in rapid changes and require fewer training epochs.

**Importance:**
The learning rate is perhaps the most important hyperparameter. If you have time to tune only one hyperparameter, tune the learning rate.
We update any weight using this equation:

						      𝑤(𝑛𝑒𝑤) = 𝑤(𝑜𝑙𝑑) − 𝐿𝑅(∂𝐸𝑟𝑟𝑜𝑟/∂𝑤(𝑜𝑙𝑑))
              
## Use of Learning Rate:
The **learning rate** hyperparameter controls the rate or speed at which the model learns. 
Specifically, it controls the amount of apportioned error that the weights of the model are updated with each time they are updated, such as at the end of each batch of training examples.
Given a perfectly configured learning rate, the model will learn to best approximate the function given available resources (the number of layers and the number of nodes per layer) in a given number of training epochs (passes through the training data).
Generally, a large learning rate allows the model to learn faster, at the cost of arriving on a sub-optimal final set of weights. A smaller learning rate may allow the model to learn a more optimal or even globally optimal set of weights but may take significantly longer to train.
At extremes, a learning rate that is too large will result in weight updates that will be too large and the performance of the model (such as its loss on the training dataset) will oscillate over training epochs. Oscillating performance is said to be caused by weights that diverge (are divergent). A learning rate that is too small may never converge or may get stuck on a suboptimal solution.

# 3. How are weights initialized?

The nodes in neural networks are composed of parameters referred to as weights used to calculate a weighted sum of the inputs.
Neural network models are fit using an optimization algorithm called **stochastic gradient descent** that incrementally changes the network weights to minimize a loss function, hopefully resulting in a set of weights for the mode that is capable of making useful predictions.

This optimization algorithm requires a starting point in the space of possible weight values from which to begin the optimization process. Weight initialization is a procedure to set the weights of a neural network to small random values that define the starting point for the optimization (learning or training) of the neural network model.

# 4. What is "loss" in a neural network?

Loss is nothing but the prediction error of the neural network.
Typically, with neural networks, we seek to minimize the error. As such, the objective function is often referred to as a cost function or a loss function and the value calculated by the loss function is referred to as simply **“loss.”**
‘Loss’ helps us to understand how much the predicted value differ from actual value.
Function used to calculate the loss is called as “Loss function”

**Types of Loss Functions:** 

Loss functions are mainly classified into two different categories that are **Classification loss** and **Regression Loss**. 

**Classification loss** is the case where the aim is to predict the output from the different categorical values for example, if we have a dataset of handwritten images and the digit is to be predicted that lies between (0-9), in these kinds of scenarios classification loss is used.
Whereas if the problem is regression like predicting the continuous values for example, if need to predict the weather conditions or predicting the prices of houses on the basis of some features. In this type of case, **Regression Loss** is used. 

# 5. What is the "chain rule" in gradient flow?
Gradient Flow Calculus is the set of rules used by the Backprop algorithm to compute gradients (this also accounts for the use of the term “flow” in tools such as Tensor Flow). Backprop works by first computing the gradients ∂l/∂yk,i≤k≤K at the output of the network (which can be computed using an explicit formula), then propagating or flowing these gradients back into the network.

Using the Chain Rule of Differentiation, the gradients ∂l/∂x and ∂l/∂y can be computed as:

		∂l/∂x=∂l/∂z.∂z/∂x
		∂l/∂y=∂l/dz.∂z/∂y

This rule can be used to compute the effect of the node on the gradients flowing back from right to left. 
