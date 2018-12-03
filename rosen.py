import numpy as np 
import sys


class Network(object):
    def __init__(self):
        self.w = np.zeros([1, N])
        self.c = 1.0
        self.learning_rate

    def r_perceptron(self):
        pass

    def train(self, features, labels, epochs, N):
        for epoch in range(epochs):
            for x, y in zip(features, labels):
                y_ = np.dot(self.w, x) * y
                # loss = 
                self.w += (1/N)*pred(y_)*x*y



def main():
    if not len(sys.argv)>=3:
        print("Incorrect amount of arguments, required: 2, optional: epochs")
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

    # generate features of dimension N with random floats sampled from a univariate normal (Gaussian) distribution of mean 0 and variance 1
    features = np.random.randn(P,N)
    # make random labels with 50/50 chance for a -1 or 1
    labels = (np.random.randint(0,2,[P])*2)-1

    # print(f"Features shape: {features.shape}")
    # print(f"Labels shape: {labels.shape}")

    




if __name__ == "__main__":
    main()