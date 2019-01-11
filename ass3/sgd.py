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
        flag = False
        if not batch:
            X = np.expand_dims(X, 1)
            Y = np.expand_dims(Y, 1)
        for idx in trange(steps):
            shuffle_in_unison(X, Y)
            if not batch:
                step_loss = []
                # for x, y in zip(X, Y):
                # for idx in range(100):
                for idx_ in range(X.shape[0]):
                    idx_ = np.random.randint(0,X.shape[0])
                    x = X[idx_]
                    y = Y[idx_]
                    loss = self.train(x, y)
                loss = self.eval(X, Y)
                losses.append(loss)
            else:
                loss = 0.5 * np.mean(np.square(self.train(X, Y)))
                losses.append(loss)
            if X_ is not None:
                losses_.append(self.eval(X_, Y_))
            if idx > 10 and np.mean([curr_loss - losses[-1] for curr_loss in losses[idx-10:idx]])<(loss/5):
                flag += 1
                if flag == 10:
                    break
            else:
                flag = 0
            # self.lr *= 0.9**(idx/50)
        print(losses[0], losses_[0])
        return losses, losses_

    def eval(self, x, y):
        sig = self.__call__(x)
        return 0.5 * np.mean(np.square(sig - y))

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

    def update(self, learning_rate):
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

    xi_train = xi[:100]
    tau_train = tau[:100]
    xi_test = xi[4900:]
    tau_test = tau[4900:]
    print(np.max(tau),np.min(tau),np.mean(tau),np.median(tau))

    shuffle_in_unison(xi_train, tau_train)

    model = Network([Dense(inputs=50, units=2, activation=tanh),
                     Dense(inputs=2, units=1, activation=None, trainable=False)])
    losses_train, losses_test = model.sgd(
        xi_train, tau_train, xi_test, tau_test, steps=steps, batch=False)


    length = len(losses_train)
    plt.figure(1)
    plt.plot(np.arange(length), losses_train, label='train')
    plt.plot(np.arange(length), losses_test, label='test')
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
