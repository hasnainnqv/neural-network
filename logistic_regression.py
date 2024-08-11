import numpy as np

class Logistic_Regression:
    def __init__(self, input_size, learning_rate=0.2, epochs=1000, initial_weights=1.0):
        self.weights = np.array([initial_weights] * (input_size + 1), dtype=float)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, logits):
        prob = 1 / (1 + np.exp(-logits))  # Using np.exp for array compatibility
        return prob

    def predict(self, X_test):
        logits = self.weights[0] + np.dot(X_test, self.weights[1:])
        prob = self.activation(logits)
        return 1 if prob >= 0.5 else 0

    def Train(self, X, y):
        X_train = np.insert(X, 0, 1, axis=1)
        for epoch in range(self.epochs):
            print(f"epoch {epoch} ========================================")
            for itr in range(len(y)):
                logits = np.dot(X_train[itr], self.weights)
                prob = self.activation(logits)
                error = y[itr] - prob

                self.weights += self.learning_rate * error * X_train[itr]

            print(f"weights after epoch {epoch}: {self.weights}")


# Example data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 1, 0, 1])

lg = Logistic_Regression(input_size=len(X[0]))
lg.Train(X, y)

# Making predictions
predictions = [lg.predict(x) for x in X]
print(predictions)
print(lg.weights)
