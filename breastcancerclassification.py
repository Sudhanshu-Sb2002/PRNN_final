import math
import warnings

import matplotlib.pyplot as plt
import medmnist.evaluator as EVALUATOR
import numpy
import numpy as np
import sklearn.metrics as metrics
from PRNN_final.LinearRegression import regression_classifier
from PRNN_final.knn import knn_naive


# includes getAUC(y_true, y_score, task),getACC(y_true, y_score, task, threshold=0.5)
# and save_results(y_true, y_score, outputpath)

def normalize_images(images: np):
    new_images = np.zeros((images.shape[0], images.shape[1] * images.shape[2]))
    for i in range(images.shape[0]):
        new_images[i] = images[i].flatten()
    new_images = new_images.astype(np.float32)
    new_images /= 255.0
    return new_images


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
    plt.plot(x_graph, AUCscores)
    plt.plot(x_graph, ACCscores)
    plt.plot(x_graph, F1scores)
    plt.xlabel(xlabel)
    plt.ylabel('Scores')
    plt.title(title)
    plt.locator_params(axis='x', nbins=10)
    plt.grid(True, which='both')
    plt.legend(['AUC', 'ACC', 'F1'])
    plt.show()


def k_means_evaluator(train_images, test_images, train_labels, test_labels, *, kmax=50):
    x_graph = numpy.zeros(kmax)
    AUCscores = numpy.zeros(x_graph.size)
    ACCscores = numpy.zeros(x_graph.size)
    F1scores = numpy.zeros(x_graph.size)
    for k in range(0, kmax):
        Y_test = knn_naive(train_images, train_labels, test_images, k=k, metric=2)
        x_graph[k] = k
        AUCscores[k] =metrics.f1_score(test_labels,Y_pred)

        ACCscores[k] = EVALUATOR.getACC(test_labels, Y_test, 'binary-class')
        F1scores[k] = f1_score(test_labels, y_pred)
    plotter(x_graph, AUCscores, ACCscores, np.zeros(x_graph.size), 'K-means binary classification', 'K')


def regressionstuff(train_images, train_labels, test_images, test_labels, km):
    x_graph = numpy.zeros(km)
    AUCscores = numpy.zeros(x_graph.size)
    ACCscores = numpy.zeros(x_graph.size)
    F1scores = numpy.zeros(x_graph.size)
    for k in range(km):
        y_pred = regression_classifier(train_images, train_labels, test_images, lambda_hyper=math.exp(-k / 10))
        for i in range(len(y_pred)):
            if y_pred[i][0] < 0.5:
                y_pred[i][0] = 0
            else:
                y_pred[i][0] = 1
        x_graph[k] = k
        AUCscores[k] = EVALUATOR.getAUC(test_labels, y_pred, 'binary-class')
        ACCscores[k] = EVALUATOR.getACC(test_labels, y_pred, 'binary-class')
        F1scores[k] = f1_score(test_labels, y_pred)
    plotter(x_graph, AUCscores, ACCscores, F1scores, 'Regression based binary classifier', '-10*log($\lambda$)')


def main(inputpath):
    train_images, val_images, test_images, train_labels, val_labels, test_labels = load_input(inputpath)

    # now we train the data using different methods
    # k_means_evaluator(train_images, test_images, train_labels,test_labels,kmax=50)
    regressionstuff(np.vstack((train_images, val_images)), np.vstack((train_labels, val_labels)), test_images,
                    test_labels, 100)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main("breastmnist.npz")
