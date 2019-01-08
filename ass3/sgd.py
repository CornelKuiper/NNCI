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

    def sgd(self, X, Y, X_=None, Y_=None, steps=500, batch=True):
        losses = []
        losses_ = []
        if not batch:
            X = np.expand_dims(X, 1)
            Y = np.expand_dims(Y, 1)
        for idx in trange(steps):
            shuffle_in_unison(X, Y)
            if not batch:
                step_loss = []
                for x, y in zip(X, Y):
                    loss = self.train(x, y)
                    step_loss.append(loss)
                losses.append(0.5 * np.mean(np.square(step_loss)))
            else:
                loss = self.train(X, Y)
                losses.append(0.5 * np.mean(np.square(loss)))
            if X_ is not None:
                losses_.append(self.eval(X_, Y_))
            # self.lr *= 0.9**(idx/50)
        print(losses[0], losses_[0])
        return losses, losses_

    def eval(self, x, y):
        sig = self.__call__(x)
        return 0.5 * np.mean(np.square(y - sig))

    def train(self, x, y):
        sig = self.__call__(x)
        loss = sig - y
        self.back_prop(loss)
        return loss


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
            return self.activation(self.y)
        return self.y

    # def __repr__(self):
    #     return f'Dense[shape:{self.w.shape}]'

    def gradients(self, loss):
        # print(f'backwards pass {self}')
        if self.activation is None:
            self.grad = loss
        else:
            self.grad = loss * self.activation(self.y, deriv=True)

        grad = np.matmul(self.grad, np.transpose(self.w))

        return grad

    def update(self, learning_rate=0.05):
        # print(f'update pass {self}')
        if self.trainable:
            grad = np.matmul(np.transpose(self.x), self.grad)
            self.w -= learning_rate * grad


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def main():
    steps = 100
    x = loadmat('data3.mat')
    xi = x['xi']
    tau = x['tau']
    xi = np.transpose(xi)
    tau = np.transpose(tau)

    shuffle_in_unison(xi, tau)

    xi_train = xi[:4500]
    tau_train = tau[:4500]
    xi_test = xi[4500:]
    tau_test = tau[4500:]
    print(np.max(tau),np.min(tau),np.mean(tau),np.median(tau))

    shuffle_in_unison(xi_train, tau_train)

    model = Network([Dense(inputs=50, units=2, activation=tanh),
                     Dense(inputs=2, units=1, activation=None, trainable=False)])
    losses_train, losses_test = model.sgd(
        xi_train, tau_train, xi_test, tau_test, steps=steps, batch=False)

    plt.figure(1)
    plt.plot(np.arange(steps), losses_train, label='train')
    plt.plot(np.arange(steps), losses_test, label='test')
    plt.title('loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(50), model.layers[0].w[:, 0])
    plt.title('w_1')
    plt.xlabel('weight')
    plt.ylabel('weight value')
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(50), model.layers[0].w[:, 1])
    plt.title('w_2')
    plt.xlabel('weight')
    plt.ylabel('weight value')
    plt.show()


if __name__ == "__main__":
    main()
