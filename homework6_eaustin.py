import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient


# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack(w):
    W1 = w[0:NUM_INPUT * NUM_HIDDEN]
    W1 = np.reshape(W1, (NUM_HIDDEN, NUM_INPUT))
    w = np.delete(w, np.arange(NUM_INPUT * NUM_HIDDEN))
    b1 = w[0:NUM_HIDDEN]
    w = np.delete(w, np.arange(NUM_HIDDEN))
    W2 = w[0:NUM_HIDDEN * NUM_OUTPUT]
    W2 = np.reshape(W2, (NUM_OUTPUT, NUM_HIDDEN))
    w = np.delete(w, np.arange(NUM_HIDDEN * NUM_OUTPUT))
    b2 = w[0:NUM_OUTPUT]
    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack(W1, b1, W2, b2):
    return np.concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("fashion_mnist_{}_images.npy".format(which))
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    images = images / 255.00
    labels = hotten(labels)
    return images, labels


def reluprime(x):
    return 1 * (x > 0)


def relu(x):
    return np.maximum(x, 0)


def hotten(x):
    return np.eye(NUM_OUTPUT)[x]


# Each row returned sum equals 1, representing probability of its the guess
# Returns a (N,10) matrix
def softmax(x):
    exp = np.exp(x)
    sum_exp = np.sum(exp, axis=0)
    return (exp / sum_exp[None, :]).T


## Function that performs transformation and gives the softmax of that
def get_yHat(X, w):
    W1, b1, W2, b2 = unpack(w)
    z1 = W1.dot(X.T) + b1[:, np.newaxis]  # (40,5)
    h = relu(z1)  # (40,5)
    z2 = W2.dot(h) + b2[:, np.newaxis]  # (10,5)
    y_Hat = softmax(z2)  # (10.5)
    return y_Hat


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE(X, Y, w):
    y_Hat = get_yHat(X, w)
    return np.sum((Y) * np.log(y_Hat)) * (-1 / Y.shape[0])


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    y_hat = get_yHat(X, w)

    y_hatT = y_hat.T
    yT = Y.T

    z1 = W1.dot(X.T) + b1[:, np.newaxis]  # (40,5)
    h = relu(z1)  # (40,5)

    grad_W2 = (y_hatT - yT).dot(h.T) / len(Y)  # (10,40)
    grad_b2 = np.mean(y_hatT - (yT), axis=1)  # (10,)

    g_T = np.multiply(((y_hatT - yT).T.dot(W2)), reluprime(z1.T))  # (N,40)

    grad_W1 = g_T.T.dot(X) / len(Y)
    grad_b1 = np.mean(g_T.T, axis=1)  # (40,1)

    gradPacked = pack(grad_W1, grad_b1, grad_W2, grad_b2)

    return gradPacked


def fPC(X, Y, w):
    y_hat = get_yHat(X, w)
    maxhat = np.argmax(y_hat, axis=1)
    maxy = np.argmax(Y, axis=1)
    acc = np.sum(1 * (maxy == maxhat)) / Y.shape[0]
    return acc


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train(trainX, trainY, testX, testY, w):
    w = SGD(trainX, trainY, w, 0.01, 256)
    print(fCE(testX, testY, w))
    print(fPC(testX, testY, w))
    return w


def SGD(x, y, w, epsilon, bactchSize):
    epoch = (x.shape[0] // bactchSize) - 1
    bactchnum = 0
    epochnum = 1000
    shuffle = np.random.permutation(y.shape[0])
    x = x[shuffle, :]
    y = y[shuffle, :]
    for e in range(0, epochnum):
        for i in range(0, epoch):
            bactchnum = bactchnum + 1
            minix = x[0 + i * bactchSize:bactchSize + i * bactchSize, :]
            miniy = y[0 + i * bactchSize:bactchSize + i * bactchSize, :]
            gradient = gradCE(minix, miniy, w)
            w = w - (epsilon * gradient)
            if bactchnum >= (epochnum * epoch) - 19:
                print(bactchnum, fCE(minix, miniy, w))
    return w


if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    idxs = [1, 2, 3, 4, 5]
    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    #print("Check grad: ",
     #     scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs, :]), np.atleast_2d(trainY[idxs, :]), w_), \
      #                              lambda w_: gradCE(np.atleast_2d(trainX[idxs, :]), np.atleast_2d(trainY[idxs, :]),
       #                                               w_), \
        #                            w))

    # Train the network using SGD.
    w = train(trainX, trainY, testX, testY, w)


