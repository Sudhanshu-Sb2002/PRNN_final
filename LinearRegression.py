import numpy as np


def leastsq1(x, y, lambda_hyper=0):
    a = transformX(x)
    cov_matrix = np.matmul(a.T, a) + lambda_hyper * np.identity(a.shape[1])
    return np.linalg.solve(cov_matrix, np.dot(a.T, y))


def transformX(x):
    return np.column_stack((x, np.ones(x.shape[0])))


def regression_classifier(X_train, y_train, X_test, *, lambda_hyper=0):
    w = leastsq1(X_train, y_train, lambda_hyper)
    y_pred = np.dot(transformX(X_test), w)
    return y_pred
