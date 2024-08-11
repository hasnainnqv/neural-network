import numpy as np
import math
class Winnow:
    def __init__(self, input_size, learning_rate=2, epochs=10, initial_weights= 0.8,threshold = 0.1):
        # self.weights = np.array([initial_weights]*(input_size))
        self.weights = np.array([0.5,0.5])
        self.learning_rate = learning_rate
        self.threshold = input_size - threshold
        self.epochs = epochs

    def activation(self, logits):
        return 1 if logits >= self.threshold else 0

    def predict(self,X_test):
        logits = np.dot(self.weights,X_test)
        return self.activation(logits)

    def Train(self,X,y):
        self.X_train = X
        self.y_train = y

        for epoch in range(self.epochs):
            print(f"epoch {epoch} ========================================")

            for itr in range(len(y)):
                logits = np.dot(self.X_train[itr],self.weights)
                predict = self.activation(logits)
                error = self.y_train[itr] - predict

                # self.weights = self.weights*self.learning_rate**(error) if self.X_train[itr]>0.3 else self.weights
                for x in range(len(self.X_train[0])):
                    updated = self.weights[x]*(math.pow(self.learning_rate, error)) if self.X_train[itr][x]>0.3 else self.weights[x]
                    self.weights[x] = updated
                print(self.weights)
            print(f"final weight of epoch {self.weights}")


X = np.array([[0.2,0.6],
              [0.3,0.9],
              [0.6,0.4],
              [0.3,0.8],
              [0.5,0.1],
              [0.8,0.3]])

y = np.array([0, 0, 1, 0, 1, 1])

winnow = Winnow(input_size=len(X[0]))
winnow.Train(X, y)

predictions = [winnow.predict(x) for x in X]
print(winnow.weights)

