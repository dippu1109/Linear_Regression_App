import numpy as np

class LinearRegression:
    def __init__(self):
        self.weight = None
        self.bias = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.weight = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        self.bias = y.mean() - self.weight * x.mean()

    def predict(self, x):
        x = np.array(x)
        if self.weight is None or self.bias is None:
            raise ValueError("Model not trained.")
        return self.weight * x + self.bias
