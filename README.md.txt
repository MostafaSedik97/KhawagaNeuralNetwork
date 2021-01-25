# NeuralNetwork

Description: We designed a configurable neural network framework that can be used in mulitple problems. The framework supports mulitple activation Fns and also multiple 
optimization algorithms as well as a comprehensive evaluation metrics. It's tested and measured using the MNIST dataset with the final model accuracy reaching over 80%.

Table Of Contents:
1-Activation Functions
2-Optimization Algorithms
3-Evaluation Metrics
4-Visualization
5-Usage

1-Activation Functions:

Implemented Activation Functions Include:
-Relu
-Tanh
-Sigmoid
-Softmax

Other Functions can be easily added and passed through the activation fns .py.

2-Optimization Algorithms:

Implemented Optimization Algorithms Include:
-Gradient Descent
-AdaGrad
-Mommentum Based
-Nesstrove
-RMSProp
-AdaDelta

3-Evaluation Metrics:

Implemented Evaluation Metrics Include:
-Confusion Matrix
-Accuracy
-Percision
-Recall
-f1 Score

4-Visualization:

-Simple plotting of any two vectors


5-Usage

a- To add a model use DeepNeuralNetwork Class passing it the needed epochs, learning rate and the optimization algorithms along as well as its parameters.
b- use the .add function to add different FC layers just by passing the size and the desired activation fn.
c- use the .addout function to add softmax layers with passing the size.
d- use the .train function to start the training process with passing the required training features and labels as well as the testing features and labels


