import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# My Functions

def reluprime(x):
    return 1 * (x > 0)

def relu(x):
    return np.maximum(x,0)

def hotten(y):
    new_Y = np.zeros((10,y.shape[0]))
    for i,val in enumerate(y):
        new_Y[val][i] = 1
    return new_Y.T

def softmax(x):
    z = np.exp(x) / np.sum(np.exp(x),axis=0)
    return z.T

def get_yHat(x,w):
    W1, b1, W2, b2 = unpack(w)
    z1 = np.dot(W1,x.T) + b1[:,None]
    h = relu(z1)    
    z2 = np.dot(W2,h) + b2[:,None]
    return softmax(z2)

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
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
def pack (W1, b1, W2, b2):
    W = W1.flatten()
    W = np.append(W, b1)
    W = np.append(W, W2.flatten())
    W = np.append(W, b2)
    return W

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which))
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    labels = hotten(labels)
    return images, labels

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    yhat = get_yHat(X, w) #(5,10)
    cost = np.sum((Y*np.log(yhat)))
    cost = cost * (-1/Y.shape[0])
    return cost

#TODO: Tweek gradCE
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    yHat = get_yHat(X,w) #(10,5)
    z1 = W1.dot(X.T) + b1[:,None]
    h = reluprime(z1)
    
    gradW2fCE = (yHat - Y).T.dot(h.T) # (10,40)
    gradb2fCE = np.mean(yHat - Y,axis=0).reshape(10,1) #(10,1)
    
    g_Trans = np.multiply(((yHat-Y).dot(W2)),reluprime(z1.T)) #(5,40)
    gradW1fCE = g_Trans.T.dot(X) # (40,785)
    gradb1fCE = np.mean(g_Trans,axis=0).reshape(40,1) # (40,1)

        
    return pack(gradW1fCE,gradb1fCE,gradW2fCE,gradb2fCE)

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w):
    pass

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Regulazing stuff
    trainX = trainX/255.00


    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))

    # Train the network using SGD.
    train(trainX, trainY, testX, testY, w)
