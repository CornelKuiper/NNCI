import numpy as np 
import sys
import matplotlib.pyplot as plt
from tqdm import trange, tqdm



class Network(object):
    def __init__(self, N):
        # self.w = np.random.randn(N) / 2
        self.w = np.zeros([N,1])
        # self.b = np.random.randn(1) / 2
        self.b = -1.0
        self.c = 0.5
        self.learning_rate = 0.1
        self.error = []

    def train_old(self, features, labels, epochs, N):
        for epoch in range(epochs):
            for x, y in zip(features, labels):
                E = ((np.dot(x, self.w) + self.b) * y)[0]
                E_ = (E < self.c).astype(float)
                self.w += (1.0 / N) * E_ * x * y
                loss += E_

            self.error.append(loss)
            if loss <= 0:
                break        
        # plt.plot(self.error)
        # plt.ylabel('error')
        # plt.show()
        return loss <= 0

    def train(self, features, labels, epochs, N):
        for epoch in range(epochs):
            loss = 0.0
            E = ((np.dot(features, self.w) + self.b) * labels)
            E_ = (E < self.c).astype(float)
            self.w += np.expand_dims(np.mean(E_ * features * labels, axis=0), -1)
            self.b += np.mean(E_ * features * labels)
            loss += np.mean(E_)


            self.error.append(loss)
            if loss <= 0:
                break        
        # plt.plot(self.error)
        # plt.ylabel('error')
        # plt.show()
        return loss <= 0



def main():
    if not len(sys.argv)>=3:
        print("Incorrect amount of arguments, required: 2, optional: epochs, nd")
        print("Give P, N and epochs")
        sys.exit(-1)
    # the amount of random vectors
    P = int(sys.argv[1])
    # N dimensions
    N = int(sys.argv[2])
    # epochs
    if len(sys.argv)>3:
        epochs = int(sys.argv[3])
    else:
        epochs = 100 
    if len(sys.argv)>4:
        nd = int(sys.argv[4])
    else:
        nd = 10


    # generate features of dimension N with random floats sampled from a univariate normal (Gaussian) distribution of mean 0 and variance 1
    features = np.random.randn(P,N)
    # make random labels with 50/50 chance for a -1 or 1
    labels = (np.random.randint(0,2,[P, 1])*2)-1

    # model = Network(N)
    # model.train(features, labels, epochs, N)

    # print(f"Features shape: {features.shape}")
    # print(f"Labels shape: {labels.shape}")
    alphas = np.arange(0.75, 3, 0.1)
    for idx_n, N in enumerate(tqdm([20, 40, 60, 80, 100])):
        runs = np.zeros_like(alphas)
        for idx_alpha, alpha in enumerate(tqdm(alphas)):
            success = []
            for run in range(nd):
                P = int(alpha*N)
                features = np.random.randn(P,N)
                # make random labels with 50/50 chance for a -1 or 1
                labels = (np.random.randint(0,2,[P, 1])*2)-1
                model = Network(N)
                success.append(model.train(features, labels, epochs, N))
            runs[idx_alpha] = np.mean(success)
        plt.plot(alphas, runs, label='N=' + str(N))
    plt.ylabel('success rate')
    plt.xlabel('alpha')
    plt.legend()
    plt.savefig('result.png')
    # plt.show()

if __name__ == "__main__":
    main()