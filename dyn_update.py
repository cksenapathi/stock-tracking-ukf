import numpy as np

class DynamicUpdate:
    def __init__(self, n):
        self.mean = np.zeros((n,1))
        self.cov = np.zeros((n,n))
        self.A = 0.1 * np.eye(n)
        self.b = np.zeros((n, 1))
        self.n = 0

    def update(self, new_data):
        new_data = np.array([new_data]).T
        self.A += np.outer(new_data, new_data)
        self.b += new_data
        new_mean = self.b/(self.n + 1)
        new_cov = (self.A - new_data @ self.b.T - self.b @ new_data.T + (self.n+1)*np.outer(new_data, new_data))/(self.n + 1)
        self.n += 1
        return (new_mean, new_cov)

    def mean_cov(self):
        return self.mean, self.cov
