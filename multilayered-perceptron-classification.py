"""

Accuracy after test: either 90%, 95% ,or 100%


"""

import numpy as np 
import random as rd
from os.path import exists

class MLP:
    def __init__ (self, input_n, hidden_n, output_n):
        print("Initializing weights...\n")

        if (self.haveSavedWeightBias()):
            self.loadWeightBias()
        else:
            self.hW = np.random.rand(input_n, hidden_n)
            self.oW = np.random.rand(hidden_n, output_n)
            self.hB = np.random.rand(hidden_n)
            self.oB = np.random.rand(output_n)
    
    def loadWeightBias(self):
        self.hW = np.genfromtxt('hW.csv', delimiter=',')
        self.hB = np.genfromtxt('hB.csv', delimiter=',')
        self.oW = np.genfromtxt('oW.csv', delimiter=',')
        self.oB = np.genfromtxt('oB.csv', delimiter=',')

    def haveSavedWeightBias(self):
        hW_exists = exists("hW.csv")
        hB_exists = exists("hB.csv")
        oW_exists = exists("oW.csv")
        oB_exists = exists("oB.csv")

        if (hW_exists and hB_exists and oW_exists and oB_exists):
            return True
        else:
            return False

    def saveWeightsBias(self):
        np.savetxt("hW.csv", self.hW, delimiter = ",")
        np.savetxt("hB.csv", self.hB, delimiter = ",")
        np.savetxt("oW.csv", self.oW, delimiter = ",")
        np.savetxt("oB.csv", self.oB, delimiter = ",")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum()

    def cce(self, t, pred_prob):
        return -np.sum(t * np.log(pred_prob))

    def forward(self, x, w, b):
        return self.sigmoid(np.dot(x, w) + b)

    def backward(self, a, b):
        return a * (1 - a) * b
    
    def updateHidden(self, lr, hG, x):
        self.hW += lr * np.outer(x, hG)
        self.hB += lr * hG

    def updateOutput(self, lr, oG, yH):
        self.oW += lr * np.outer(yH, oG)
        self.oB += lr * oG

    def predict(self, x):
        hY = self.forward(x, self.hW, self.hB)
        oY = self.forward(hY, self.oW, self.oB)
        pred_prob = self.softmax(oY)

        max = np.max(pred_prob)
        for i in range(len(pred_prob)):
            if pred_prob[i] == max:
                pred_prob[i] = 1
            else:
                pred_prob[i] = 0
        return pred_prob
    
    def accuracy_score(self, acc_pred, pred):
        return (acc_pred / pred) * 100

    def train_test_split(self, n, X, Y):
        arr = np.array([i for i in range(n)])
        rd.shuffle(arr)

        new_X = np.array([X[i] for i in arr])
        new_Y = np.array([Y[i] for i in arr])

        train_len = int(np.round(n * 0.8, 0))

        # Selects from 0 to train length value for training data
        # Selects from training data length to max length for the testing data
        return new_X[:train_len], new_Y[:train_len], new_X[train_len:], new_Y[train_len:]

    def fit(self, X, Y, lr, max_epoch, threshold):
        print(f"Train the model with LR: {lr} Max Epoch: {max_epoch} Threshold: {threshold}...\n")

        epoch = 0

        while epoch < max_epoch:
            total_error = 0
            acc_pred = 0
            pred = 0

            for x, t in zip(X, Y):
                hY = self.forward(x, self.hW, self.hB)
                oY = self.forward(hY, self.oW, self.oB)

                oG = self.backward(oY, (t - oY))
                hG = self.backward(hY, np.dot(oG, self.oW.T))

                # Evaluating error by running the predicted y's into
                # a softmax activation function to get each values' probability of being the actual y
                pred_prob = self.softmax(oY)

                self.updateOutput(lr, oG, hY)
                self.updateHidden(lr, hG, x)

                total_error += self.cce(t, pred_prob)

                if np.argmax(t) == np.argmax(pred_prob):
                    acc_pred += 1
                pred += 1

            total_error /= X.size

            epoch += 1
    
            acc_score= self.accuracy_score(acc_pred, pred)

            print(f"Epoch: {epoch} Total Error: {total_error:.4f} Overall Accuracy: {acc_score:.2f}%")

            if acc_score >= 95:
                self.saveWeightsBias()

            if total_error <= threshold: 
                print("Error Threshold has been reached. Training stopped.\n")
                break
            if epoch == max_epoch: 
                print("Max epochs reached. Training Stopped.\n")


# importing the dataset
dataset = np.genfromtxt('dataset/family_wealth_dataset.csv', delimiter=',', skip_header=1)
# dataset = np.unique(dataset, axis=0)

# Getting the X and Y's 
X = dataset[:, :-5]
Y = dataset[:, -5:]

input_n = X[0].size


# Customization Area
hidden_n = 8
output_n = 5

lr = 0.01 
max_epoch = 5000
threshold = 0.0001


mlp = MLP(input_n, hidden_n, output_n)


# Splitting the dataset
n_row = X[:, 0].size
x_train, y_train, x_test, y_test = mlp.train_test_split(n_row, X, Y)


# Trains model if weights are not yet saved and save it after training
if (not mlp.haveSavedWeightBias()):
    mlp.fit(x_train, y_train, lr, max_epoch, threshold)


# Testing the MLP
print("Testing the MLP...")

acc_pred = 0
pred = 0

for x, y_actual in zip(x_test, y_test):
    y_pred = mlp.predict(x)

    if np.argmax(y_actual) == np.argmax(y_pred):
        acc_pred += 1
    pred += 1

    print(f"Given x: {x} Actual: {y_actual} Prediction: {y_pred}")

print(f"Correct Predictions: {acc_pred}")
print(f"Number of inputs: {len(x_test)}")
print(f"Accuracy: {mlp.accuracy_score(acc_pred, pred):.2f}%")