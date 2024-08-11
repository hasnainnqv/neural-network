import math

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# inputs = [1, 2, 3, 2.5]
# weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-.26,-0.27,0.17,0.87]]
# biases = [2,3,0.5]
# outputs = np.dot(weights,inputs) + biases
# print(outputs)

# X = [[1,-3,5,1],
#      [4,-1,6,1],
#      [4,-7,9,9],
# ]

class DenseLayer:
    def __init__(self,inputs, neurons):
        self.weights = np.random.randn(inputs, neurons)
        self.bias =np.zeros((1, neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class ActivationRelu:
    def forward(self, inputs):
        self.output = 0.1*np.maximum(0,inputs)
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, keepdims=True, axis=1)

class Loss:
    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs,y)
        data_loss = np.mean(sample_losses)
        return data_loss
class LossCrossEntropy(Loss):
    def forward(self, y_pred, y):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7)
        if len(y.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y, axis = 1)
        negative_log = -np.log(correct_confidence)
        return negative_log




X,y = spiral_data(1000,5)
layer1 = DenseLayer(2,5)
layer2 = DenseLayer(5,5)
activation1 = ActivationRelu()
activation2 = ActivationSoftmax()
# plt.scatter(X[:,0],X[:,1], s=20,c = y, cmap='brg')
# plt.show()
loss = LossCrossEntropy()

lowest_loss = 9999999
best_dense1_weight =  layer1.weights.copy()
best_dense1_biases =  layer1.bias.copy()
best_dense2_weight =  layer2.weights.copy()
best_dense2_biases =  layer2.bias.copy()

for iterations in range(1000):
    layer1.weights += 0.05*np.random.randn(2,5)
    layer1.bias += 0.05*np.random.randn(1,5)
    layer2.weights += 0.05*np.random.randn(5,5)
    layer2.bias += 0.05*np.random.randn(1,5)

    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss_function = loss.calculate(activation2.output, y)

    predications = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predications == y)

    if loss_function < lowest_loss:
        print(f"New set of weights found, iteration {iterations} loss: {loss_function} accuracy: {accuracy}")
        best_dense1_weight = layer1.weights.copy()
        best_dense1_biases = layer1.bias.copy()
        best_dense2_weight = layer2.weights.copy()
        best_dense2_biases = layer2.bias.copy()
        lowest_loss = loss_function

    else:
        layer1.weights = best_dense1_weight.copy()
        layer1.bias = best_dense1_biases.copy()
        layer2.weights = best_dense2_weight.copy()
        layer2.bias = best_dense2_biases.copy()