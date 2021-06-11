try:  # if numpy is not found
    import numpy as np
except Exception as e:
    print(e)
    print("Install numpy with 'pip3 install numpy' command (or pip)")
    exit(1)

class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x + bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self.sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * 2 * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * 2 * np.sum(y_predicted - y)
            # update parameters according to cost
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear)
        return y_predicted  #returning the probability

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
