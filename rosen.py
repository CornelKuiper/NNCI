import numpy as np 
import sys


class Network(object):
    def __init__(self):
        self.w = 0
        

    def r_perceptron(self):
        pass



def main():
    if not len(sys.argv)==3:
        print("Incorrect amount of arguments, required: 2")
        print("Give P and N")
        sys.exit(-1)
    # the amount of random vectors
    P = int(sys.argv[1])
    # N dimensions
    N = int(sys.argv[2])

    # generate features of dimension N with random floats sampled from a univariate normal (Gaussian) distribution of mean 0 and variance 1
    features = np.random.randn(P,N)
    # make random labels with 50/50 chance for a -1 or 1
    labels = (np.random.randint(0,2,[P])*2)+1

    # print(f"Features shape: {features.shape}")
    # print(f"Labels shape: {labels.shape}")




if __name__ == "__main__":
    main()