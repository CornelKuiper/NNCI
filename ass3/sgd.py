import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import trange, tqdm


class Network(object):
    def __init__(self, N):
        self.w = np.random.rand(N, 2)
        self.w /= np.sum(self.w, 0, keepdims=True)

        self.lr = 0.05


    def sgd_(self, x, y):
        y_ = np.tanh(np.dot(x, self.w))
        y_ = np.sum(y_, 1, keepdims=True)
        loss = 0.5 * np.mean(np.square(y_ - y))
        e = 1

        self.w -= 1

    def sgd(self, x, y, steps=1000):
        P = x.shape[0]  * 1.0
        idx_ = np.random.randint(P, size=steps)
        loss = []
        # for x_, y_ in zip(x, y):
        for idx in idx_:
            x_, y_ = x[idx], y[idx]
            sig = np.tanh(np.dot(x_, self.w))
            sig = np.sum(sig)
            e_v = 0.5 * np.square(sig - y_)
            loss.append((1 / P) * e_v)
            self.w -= self.lr * e_v
        plt.plot(loss)
        plt.show()




def main():
    x = loadmat('data3.mat')
    xi = x['xi']
    tau = x['tau']
    print(xi.shape)
    print(tau.shape)
    xi = np.transpose(xi)
    tau = np.transpose(tau)
    N = xi.shape[-1]
    model = Network(N)
    model.sgd(xi, tau)
    print(sum(xi[0]))

if __name__ == "__main__":
    main()