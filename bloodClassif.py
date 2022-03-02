import math

import matplotlib.pyplot as plt
import numpy
import numpy as np

from PRNN_final.LinearRegression import regression_classifier
from PRNN_final.knn import knn_naive


# includes getAUC(y_true, y_score, task),getACC(y_true, y_score, task, threshold=0.5)
# and save_results(y_true, y_score, outputpath)

def normalize_images(images: np):
    new_images = np.zeros((images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]))
    for i in range(images.shape[0]):
        new_images[i] = images[i].flatten()
    new_images = new_images.astype(np.float32)
    new_images /= 255.0
    return new_images


def load_input(inputpath):
    folder = np.load(inputpath)
    files = folder.files
    # print(files)
    # First we load the images and the labels
    un_train_images = folder['train_images.npy']
    un_val_images = folder['val_images.npy']
    un_test_images = folder['test_images.npy']
    train_labels = folder['train_labels.npy']
    val_labels = folder['val_labels.npy']
    test_labels = folder['test_labels.npy']
    # Then we normalize the images
    train_images = normalize_images(un_train_images)
    val_images = normalize_images(un_val_images)
    test_images = normalize_images(un_test_images)
    return train_images, val_images, test_images, train_labels, val_labels, test_labels


def plotter(x_graph, AUCscores, ACCscores, F1scores, title, xlabel):
    print(x_graph, ACCscores, AUCscores, F1scores)
    if AUCscores is not None:
        plt.plot(x_graph, AUCscores, label='AUC')
    if AUCscores is not None:
        plt.plot(x_graph, ACCscores)
    if F1scores is not None:
        plt.plot(x_graph, F1scores)
    plt.xlabel(xlabel)
    plt.ylabel('Scores')
    plt.title(title)
    plt.locator_params(axis='x', nbins=10)
    plt.grid(True, which='both')

    plt.show()


def f1_score(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_true[i] == 1:
                fn += 1
            else:
                fp += 1
    if tp == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)


def one_hot_encoding(labels):
    one_hot_labels = numpy.zeros((labels.shape[0], 8))
    for i in range(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels


def regressionstuff(train_images, train_labels0, test_images, test_labels0, km):
    train_labels = one_hot_encoding(train_labels0)
    test_labels = one_hot_encoding(test_labels0)
    x_graph = numpy.zeros(km)
    AUCscores = numpy.zeros(x_graph.size)
    ACCscores = numpy.zeros(x_graph.size)
    F1scores = numpy.zeros(x_graph.size)
    for k in range(km):
        y_predonehot = regression_classifier(train_images, train_labels, test_images, lambda_hyper=-0.01*math.exp(k*10))

        y_pred = np.argmax(y_predonehot, axis=1)
        x_graph[k] = k
        ACCscores[k] = basic_classification_accuraccy(test_labels0, y_pred)
    plotter(x_graph, None, ACCscores, None, 'Regression', 'K')


def basic_classification_accuraccy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def k_means_evaluator(train_images, test_images, train_labels, test_labels, *, kmax=5):
    x_graph = numpy.zeros(kmax)

    ACCscores = numpy.zeros(x_graph.size)

    for k in range(0, kmax):
        Y_test = knn_naive(train_images, train_labels, test_images, k=k, metric=2)
        x_graph[k] = k
        # AUCscores[k] = EVALUATOR.getAUC(test_labels, Y_test, 'binary-class')
        # ACCscores[k] = EVALUATOR.getACC(test_labels, Y_test, 'binary-class')
        ACCscores[k] = basic_classification_accuraccy(test_labels, Y_test)

    plotter(x_graph, None, ACCscores, None, 'K-means based multiclass classifier', 'k')


def main(inputpath):
    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_input(inputpath)

    # now we train the data using different methods
    # k_means_evaluator(train_images, test_images, train_labels, test_labels, kmax=10)

    regressionstuff(train_images, train_labels, test_images, test_labels, km=5)


if __name__ == '__main__':
    main("bloodmnist.npz")
