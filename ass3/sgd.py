import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import trange


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

    def sgd(self, x, y, steps=1000):
        losses = []
        for idx in trange(steps):
            sig = self.__call__(x)
            loss = y - sig
            self.back_prop(loss)
            losses.append(0.5 * np.mean(np.square(loss)))
            # self.lr *= 0.9**(idx/50)
        plt.plot(losses)
        plt.show()


class Dense(object):
    def __init__(self, inputs=50, units=2, activation=tanh,
                 trainable=True):
        self.trainable = trainable
        if self.trainable:
            self.w = np.random.rand(inputs, units)
            self.w /= np.sum(np.square(self.w), 0, keepdims=True)
        else:
            self.w = np.ones([inputs, units])

        self.activation = activation
        self.y = None
        self.x = None
        self.grad = None

    def __call__(self, x):
        self.x = x
        self.y = np.matmul(self.x, self.w)
        if self.activation is not None:
            self.y = self.activation(self.y)
        return self.y

    def __repr__(self):
        return f'Dense[shape:{self.w.shape}]'

    def gradients(self, loss):
        # print(f'backwards pass {self}')
        if self.activation is None:
            self.grad = loss
        else:
            self.grad = loss * self.activation(self.y, deriv=True)
        grad = np.matmul(self.grad, np.transpose(self.w))
        return grad

    def update(self, learning_rate):
        # print(f'update pass {self}')
        if self.trainable:
            grad = np.matmul(np.transpose(self.x), self.grad)
            self.w += learning_rate * grad


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def main():
    x = loadmat('data3.mat')
    xi = x['xi']
    tau = x['tau']
    xi = np.transpose(xi)
    tau = np.transpose(tau)

    shuffle_in_unison(xi, tau)

    model = Network([Dense(inputs=50, units=2, activation=tanh),
                     Dense(inputs=2, units=1, activation=None, trainable=False)])
    model.sgd(xi, tau)


if __name__ == "__main__":
    main()
