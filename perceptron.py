import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100, threshold = 0.5):
        self.weights = np.zeros(input_size)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights) - self.threshold
        return self.activation_function(z)

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]  # Update weights

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

predictions = [perceptron.predict(x) for x in X]
print(predictions)
print(perceptron.weights)

