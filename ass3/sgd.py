import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import trange, tqdm


def tanh(x, deriv=False):
    if deriv:
        return 1.0 - np.tanh(x)**2
    return np.tanh(x)


class Network(object):
    def __init__(self, layers):
        self.layers = layers
        print(self.layers)

        self.lr = 0.05

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def back_prop(self, loss):
        for layer in self.layers[::-1]:
            # print(loss)
            loss = layer.gradients(loss)

        for layer in self.layers:
            layer.update(self.lr)


    def sgd(self, x, y, steps=2000):
        losses = []
        # x = x[0:10]
        # y = y[0:10]
        # for x_, y_ in zip(x, y):
        for idx in trange(steps):
            sig = self.__call__(x)
            # print(sig.shape)
            # print(y.shape)
            # print(sig.shape, y.shape)
            # loss = 0.5 * np.mean(np.square(sig - y))
            loss = sig - y
            # print(loss)
            self.back_prop(loss)
            losses.append(0.5 * np.mean(np.square(sig - y)))
            # self.lr *= 0.95**(idx/500)

        plt.plot(losses)
        plt.show()


class Dense(object):
    def __init__(self, inputs=50, units=2, activation=tanh,
                 trainable=True):
        self.trainable = trainable
        if self.trainable:
            self.w = np.random.rand(1, inputs, units)
            self.w /= np.sum(np.square(self.w), 0, keepdims=True)
        else:
            self.w = np.ones([1, inputs, units])
        if activation is None:
            self.activation = None
        else:
            self.activation = activation
        self.y = None
        self.x = None
        self.grad = None

    def __repr__(self):
        return f'Dense[shape:{self.w.shape}]'

    def __call__(self, x):
        self.x = x
        # print('forward pass')
        self.y = np.matmul(self.x, self.w)
        # self.y = np.transpose(self.y, [1,0,2])
        if self.activation is not None:
            self.y = self.activation(self.y)

        # print(self.x.shape, self.w.shape, self.y.shape)
        return self.y

    def gradients(self, loss):
        # print(self.__repr__)
        batch_size = loss.shape[0]
        loss = np.transpose(loss, [0,2,1])
        # print(f'backwards pass {self}')
        self.w
        if self.activation is None:
            # print(self.w.shape, loss.shape, self.y.shape)
            self.grad = self.w * loss
        else:
            # print(self.w.shape, loss.shape, self.y.shape)
            self.grad = (self.w * loss) * self.activation(self.y, deriv=True)
        # print(np.mean(self.grad, 0, keepdims=True).shape)
        # self.grad = np.mean(self.grad, 0, keepdims=True)
        # print(self.w.shape)
        # print(self.y.shape)
        # print(self.grad.shape)
        return self.grad

    def update(self, learning_rate=0.05):
        if self.trainable:
            self.w -= learning_rate * np.mean(self.grad * np.transpose(self.x, [0,2,1]), 0)

def main():
    x = loadmat('data3.mat')
    xi = x['xi']
    tau = x['tau']
    # xi = np.transpose(xi)
    # tau = np.transpose(tau)
    xi = np.expand_dims(np.transpose(xi), 1)
    tau = np.expand_dims(np.transpose(tau), 1)
    N = xi.shape[-1]
    model = Network([Dense(inputs=50, units=2, activation=tanh),
                     Dense(inputs=2, units=1, activation=None, trainable=False)])
    model.sgd(xi, tau)

if __name__ == "__main__":
    main()