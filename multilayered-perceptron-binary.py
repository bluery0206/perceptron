import numpy as np 

class MLP:
    def __init__ (self, input_n, hidden_n, output_n):
        self.hW = np.random.rand(input_n, hidden_n)
        self.oW = np.random.rand(hidden_n, output_n)

        self.hB = np.random.rand(hidden_n)
        self.oB = np.random.rand(output_n)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        hY = self.sigmoid(np.dot(x, self.hW) + self.hB)
        oY = self.sigmoid(np.dot(hY, self.oW) + self.oB)
        return oY

    def updateHidden(self, lr, hG, x):
        self.hW += lr * np.outer(x, hG)
        self.hB += lr * hG

    def updateOutput(self, lr, oG, yH):
        self.oW += lr * np.outer(yH, oG)
        self.oB += lr * oG

    def predict(self, x):
        oY = self.forward(x)
        return 0 if oY < 0.5 else 1

    def backward(self, x, t, oY):
        oG = oY * (1 - oY) * (t - oY)
        hY = self.sigmoid(np.dot(x, self.hW) + self.hB)
        hG = (hY * (1 - hY)) * np.dot(oG, self.oW.T)
        return [hG, oG, hY]
    
    def fit(self, X, Y, lr, max_epoch, threshold):
        print(f"Train the model with LR: {lr} Max Epoch: {max_epoch} Threshold: {threshold}...")

        epoch = 0

        while epoch < max_epoch:
            total_error = 0

            for x, t in zip(X, Y):
                oY = self.forward(x)

                hGoGhY = self.backward(x, t, oY)
                hG = hGoGhY[0]
                oG = hGoGhY[1]
                hY = hGoGhY[2]

                self.updateOutput(lr, oG, hY)
                self.updateHidden(lr, hG, x)
                
                total_error += ((t - oY)**2) / len(X)
            
            print(f"Epoch: {epoch} Total Error: {total_error}")

            epoch += 1

            if total_error <= threshold: 
                print("Error Threshold has been reached. Training stopped.\n")
                break
            elif epoch == max_epoch: 
                print("Max epochs reached. Training Stopped.\n")


dataset = np.genfromtxt('dataset/alphabet.csv', delimiter=',', skip_header=1)

X = dataset[:, :-1]
Y = dataset[:, -1]

input_n = 35
output_n = 1
hidden_n = round(input_n / 2) + 1

lr = 0.1
max_epoch = 10000
threshold = 0.001    

mlp = MLP(input_n, hidden_n, output_n)
mlp.fit(X, Y, lr, max_epoch, threshold)


# RANDOMIZED SELECTION OF INPUTS
arr = [x for x in range(dataset[:, 0].size- 1)]
np.random.shuffle(arr)

print("Testing the MLP...")
for x in arr:
    # Converts each input into a string and implodes them into a string.
    stringed_x = ' '.join(str(int(val)) for val in X[x])
    print(f"Given x: {stringed_x} Actual: {int(Y[x])} Prediction: {mlp.predict(X[x])}")

# LINEAR SELECTION
# for x, y_actual in zip(X, Y):
#     # Converts each input into a string and implodes them into a string.
#     stringed_x = ' '.join(str(int(val)) for val in x)
#     print(f"Given x: {stringed_x} Actual: {int(y_actual)} Prediction: {mlp.predict(x)}")


# Getting the values of weights and biases where the prediction is accurate
# print("Hidden Weight: ", mlp.hW)
# print("Hidden Bias: ", mlp.hB)
# print("Output Weight: ", mlp.oW)
# print("Output Bias: ", mlp.oB)