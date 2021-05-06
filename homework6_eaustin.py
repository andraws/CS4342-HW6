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
    W = W1.flatten()
    W = np.append(W, b1)
    W = np.append(W, W2.flatten())
    W = np.append(W, b2)
    return W


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("fashion_mnist_{}_images.npy".format(which)) / 225
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    labels = hoten(labels)
    return images, labels


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def yhateq(x, w):
    W1, b1, W2, b2 = unpack(w)
    z1 = W1.dot(x.T) + b1[:, np.newaxis]  # (40,5)
    h = relu(z1)  # (40,5)
    z2 = W2.dot(h) + b2[:, np.newaxis]  # (10,5)
    y_Hat = softmax(z2)  # (10.5)
    return y_Hat


def fCE(X, Y, w):
    yhat = yhateq(X, w)
    return np.sum((Y) * np.log(yhat)) * (-1 / Y.shape[0])


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w):
    y_hat = yhateq(X, w)

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


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train(trainX, trainY, testX, testY, w):
    pass


def relu(x):
    return 1 * (x > 0)


def reluprime(x):
    return np.maximum(x, 0)


def softmax(x):
    exp = np.exp(x)
    sum_exp = np.sum(exp, axis=0)
    return (exp / sum_exp[None, :]).T

def hoten(y):
    return np.eye(NUM_OUTPUT)[y]


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

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    approx_fprime = scipy.optimize.approx_fprime(w, lambda W_: fCE(trainX[idxs, :], trainY[idxs, :], W_), 1e-8)
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs, :]), np.atleast_2d(trainY[idxs, :]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs, :]), np.atleast_2d(trainY[idxs, :]),
                                                      w_), \
                                    w))

    # Train the network using SGD.
    #train(trainX, trainY, testX, testY, w)
