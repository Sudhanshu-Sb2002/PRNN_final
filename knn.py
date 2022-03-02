import numpy as np


def k_smallest(arr, k):
    x = []
    min = np.inf
    pos_min = -1
    for i in range(k):
        for i in range(len(arr)):
            if arr[i] < min and i not in x:
                min = arr[i]
                pos_min = i
        if pos_min != -1:
            x.append(pos_min)
            min = np.inf
            pos_min = -1
    return x


def knn_naive(X_train, Y_train, X_test, *, k=5, metric=2):
    Y_test = np.zeros((X_test.shape[0], 1))
    if k == 0:
        return Y_test
    for i in range(X_test.shape[0]):
        distances = np.linalg.norm(np.abs(X_train - X_test[i, :]), ord=metric, axis=1)

        nearest = k_smallest(distances, k)
        topk_y = [i[0] for i in Y_train[nearest[:k]]]
        Y_test[i] = [np.argmax(np.bincount(topk_y))]
    return Y_test
