import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import trange, tqdm


class Network(object):
    def __init__(self, N):
        self.w = np.random.rand(N, 2)
        self.w /= np.sum(self.w**2, 0, keepdims=True)
        self.lr = 0.05

    def der_tanh(self, x):
        return 1 - x ** 2

    def sgd_(self, x, y):
        y_ = np.tanh(np.dot(x, self.w))
        y_ = np.sum(y_, 1, keepdims=True)
        loss = 0.5 * np.mean(np.square(y_ - y))
        e = 1
        self.w -= 1

    def sgd(self, x, y, steps=5000, epochs=100):
        P = x.shape[0]  * 1.0
        loss = []
        for epoch in trange(epochs):
            idx_ = np.random.randint(P, size=steps)
            for idx in idx_:
                x_, y_ = x[idx], y[idx]
                dot = np.dot(x_, self.w)
                sig = np.tanh(dot)
                E = sig - y_
                output = np.sum(sig)
                e_v = 0.5 * np.square(output - y_)
                # print(dot)
                # print((E[0] * self.der_tanh(dot[0]) * dot).shape)
                gradient0 = (E[0] * self.der_tanh(dot[0]) * x_) .reshape(50,1)
                gradient1 = (E[1] * self.der_tanh(dot[1]) * x_).reshape(50,1)
                gradient = np.concatenate([gradient0,gradient1],axis=1)
                self.w -= self.lr * gradient * e_v

            sig = np.tanh(np.dot(x, self.w))
            sig = np.sum(sig,axis=1).reshape(5000,1)
            e_v = np.mean(0.5 * np.square(sig - y))
            loss.append(e_v)

        plt.plot(loss)
        plt.show()
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.plot(self.w[:,0])
        ax2.plot(self.w[:,1])
        plt.show()




def main():
    x = loadmat('data3.mat')
    xi = x['xi']
    tau = x['tau']
    xi = np.transpose(xi)
    tau = np.transpose(tau)
    N = xi.shape[-1]
    model = Network(N)
    model.sgd(xi, tau)

if __name__ == "__main__":
    main()