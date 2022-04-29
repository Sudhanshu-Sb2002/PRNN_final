import matplotlib.pyplot as plt
import numpy as np


def logisticregression(train_x, train_y, test_x, *, learning_rate=0.05, epochs=5000,lambdap=0):
    # initialize the weights
    weights = np.zeros((train_x.shape[1],train_y.shape[1]))
    pred = np.zeros(train_y.shape)
    bias = np.zeros(train_y.shape[1])
    # initialize the cost_list
    costs = np.zeros(epochs)
    for iteration in range(epochs):
        # forward propagation
        pred = sigmoid(train_x.dot(weights)+bias)
        # calculate the cost
        cost = np.sum(loss(pred, train_y))
        costs[iteration] = cost
        # calculate the gradient
        m = train_x.shape[0]
        grad = train_x.T.dot(pred - train_y)/m
        bias = bias + learning_rate * (np.sum(train_y - pred))/m
        # update the weights
        weights -= learning_rate * grad

    plotter(costs, costs.size)
    return sigmoid(test_x.dot(weights)+bias)


def plotter(cost, iterations):
    plt.plot(range(iterations), cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title("Cost function decreasing as iterations increase")
    plt.grid(True, which='major')
    plt.grid(True, which='minor', linestyle='--', linewidth=0.2)
    plt.minorticks_on()
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    a = sigmoid(x)
    return a * (1 - a)


def loss(pred, y):
    if( y.size==pred.size):
        return -np.matmul(np.log(pred), y.T) + np.matmul(np.log(1 - pred), (1 - y).T)
