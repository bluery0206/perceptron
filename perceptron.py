
# Author: Mark Ryan Hilario BSCS 3-A
# Date: 13/03/2024

import numpy as np
import random as rd

class Perceptron():
    def __init__(self, size) -> None:
        self.weights = np.random.uniform(0, 1, size)
        self.bias = 1
    
    def update(self, learning_rate, row, y_actual, y_pred):
        for x in range(len(row)):
            self.weights[x] += (learning_rate * (y_actual - y_pred) * row[x])
        self.bias += (learning_rate * (y_actual - y_pred))
    
    def activation(self, sum):
        return 1 if sum > 0 else 0

    def y_predict(self, row):
        return self.activation(sum([(row * self.weights[idx]) + self.bias for idx, row in enumerate(row)]))

    def fit(self, X, Y, learning_rate, max_epoch, threshold):
        epoch = 0
        while epoch < max_epoch:
            print(f"{epoch=}")

            rand_arr = [_ for _ in range(len(X))]
            rd.shuffle(rand_arr)

            error = 0

            for i in rand_arr:
                y_pred = self.y_predict(X[i])
                y_actual = Y[i]

                if y_pred == y_actual: continue 

                self.update(learning_rate, X[i], y_actual, y_pred)
                error += abs(y_actual - y_pred)

                print(f"{y_actual=}, {y_pred=}")

            error /= len(X[i])

            print(f"{epoch=}, {error=}")

            epoch += 1

            if error <= threshold:
                print("Reached threshold... Training stopped.")
                return
            elif epoch == max_epoch:
                print("Reached max epoch... Training stopped.")
                return

X = [[1, 0], [0, 1], [0, 0], [1, 1]]
Y = [1, 0, 1, 1]

learning_rate = 0.1
max_epoch = 100
threshold = 0.01
size = len(X)

perceptron = Perceptron(size)
perceptron.fit(X, Y, learning_rate,  max_epoch, threshold)

for i in range(len(X)): print(f"Inputs: {X[i]} y_Actual: {Y[i]} y_Predicted: {perceptron.y_predict(X[i])}")
