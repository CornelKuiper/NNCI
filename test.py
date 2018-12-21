from multiprocessing import Pool
import numpy as np 
import sys
import matplotlib.pyplot as plt
from tqdm import trange, tqdm



def network(features, labels, epochs, N):
    w = np.zeros([1, N])
    b = 0.0
    c = 0.0
    w = np.append(w, b)

    features = np.insert(features, N, -1.0, axis=1)
    for epoch in range(epochs):
        loss = 0.0
        for x, y in zip(features, labels):
            E = (np.dot(w, x) * y)[0]
            E_ = (E <= c).astype(float)
            w += (1.0 / N) * E_ * x * y
            loss += E_
        if loss <= 0:
            break
    return loss <= 0

def main():
    if not len(sys.argv)>=3:
        print("Incorrect amount of arguments, required: 2, optional: epochs, nd")
        print("Give P, bias/nobias/both and epochs")
        sys.exit(-1)
    # the amount of random vectors
    P = int(sys.argv[1])
    # N dimensions
    bias = sys.argv[2]
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
    # features = np.random.randn(P,N)
    # make random labels with 50/50 chance for a -1 or 1
    # labels = (np.random.randint(0,2,[P, 1])*2)-1

    alphas = np.arange(0.75, 3, 0.05)
    
    if bias == 'bias' or bias == 'both':
        for idx_n, N in enumerate(tqdm([20, 60, 100])):
            runs = np.zeros_like(alphas)
            for idx_alpha, alpha in enumerate(tqdm(alphas)):
                inputs = [[np.random.randn(P,N), (np.random.randint(0,2,[P, 1])*2)-1, epochs, N] for run in range(nd)]
                with Pool(4) as p:
                    success = p.starmap(network, inputs)
                # print(success)
                # exit()
                runs[idx_alpha] = np.mean(success)
            plt.plot(alphas, runs, label='N=' + str(N))


    savename = str(epochs) + '_' + str(nd) + bias + '_c0.2' '.png'
    plt.ylabel('success rate')
    plt.xlabel('alpha')
    plt.legend()
    plt.savefig(savename)
    # plt.show()

if __name__ == "__main__":
    main()