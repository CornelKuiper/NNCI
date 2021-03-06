import numpy as np 
import sys
import matplotlib.pyplot as plt
from tqdm import trange, tqdm



class Network(object):
    def __init__(self, N):
        # self.w = np.random.randn(N) / 2
        self.w_batch = np.zeros([N,1])
        self.w = np.zeros([1,N])
        # self.b = np.random.randn(1) / 2
        self.b = 1.0
        self.c = 0.0
        self.learning_rate = 0.1
        self.error = []
        # self.w = np.append(self.w, self.b)

    def train(self, features, labels, epochs, N):
        # features = np.insert(features,N,-1.0,axis=1)    
        for epoch in range(epochs):
            loss = 0.0
            for x, y in zip(features, labels):
                E = (np.dot(self.w,x) * y)[0]
                E_ = (E <= self.c).astype(float)
                self.w += (1.0 / N) * E_ * x * y
                loss += E_
            self.error.append(loss)
            if loss <= 0:
                break        
        return loss <= 0

    def train_batch(self, features, labels, epochs, N):
        for epoch in range(epochs):
            loss = 0.0
            E = ((np.dot(features, self.w_batch) + self.b) * labels)
            E_ = (E <= self.c).astype(float)
            self.w_batch += np.expand_dims(np.mean(E_ * features * labels, axis=0), -1)
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

    alphas = np.arange(0.75, 5, 0.05)
    for idx_n, N in enumerate(tqdm([20, 60, 100])):
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

    savename = str(epochs) + '_' + str(nd) + '.png'
    plt.ylabel('success rate')
    plt.xlabel('alpha')
    plt.legend()
    plt.savefig(savename)
    # plt.show()

if __name__ == "__main__":
    main()