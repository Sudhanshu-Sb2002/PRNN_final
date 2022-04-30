import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics


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


def one_hot_encoding(labels):
    one_hot_labels = np.zeros((labels.shape[0], 8))
    for i in range(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels


def plotter(x_graph, ACCscores, F1scores, title, xlabel, *, first=False):
    plt.plot(x_graph, ACCscores, alpha=0.5)
    plt.plot(x_graph, F1scores, alpha=0.5)

    plt.scatter(x_graph, ACCscores, s=6)
    plt.scatter(x_graph, F1scores, s=6)

    plt.xlabel(xlabel)

    plt.title(title)
    plt.ylim(0.5, 1)
    plt.grid(True, which='major')
    plt.grid(True, which='minor', linestyle='--', linewidth=0.2)
    plt.minorticks_on()

    if first:
        plt.figlegend(['ACC', 'F1'])

path="bloodmnist.npz"
train_images, val_images, test_images, train_labels, val_labels, test_labels = load_input(path)
train_labels1 = one_hot_encoding(train_labels)
test_labels1 = one_hot_encoding(test_labels)

k_max=10

x_graph = np.zeros(k_max)
AUCscores = np.zeros(x_graph.shape)
ACCscores = np.zeros(x_graph.shape)
F1scores = np.zeros(x_graph.shape)

def leastsq1(x, y, lambda_hyper=0.0):
    a = transformX(x)
    cov_matrix = np.matmul(a.T, a) + lambda_hyper * np.identity(a.shape[1])
    return np.linalg.solve(cov_matrix, np.dot(a.T, y))


def transformX(x):
    return np.column_stack((x, np.ones(x.shape[0])))


def regression_classifier(X_train, y_train, X_test, *, lambda_hyper=0.0):
    w = leastsq1(X_train, y_train, lambda_hyper)
    y_pred = np.dot(transformX(X_test), w)
    return y_pred
def plotter(x_graph, ACCscores, F1scores, title, xlabel,*,first=False):


    plt.plot(x_graph, ACCscores,alpha=0.5 )
    plt.plot(x_graph, F1scores,alpha=0.5)

    plt.scatter(x_graph, ACCscores,s=6)
    plt.scatter(x_graph, F1scores,s=6)

    plt.xlabel(xlabel)

    plt.title(title)
    plt.ylim(0.5,1)
    plt.grid(True, which='major')
    plt.grid(True, which='minor',linestyle='--',linewidth=0.2)
    plt.minorticks_on()

    if first:
        plt.figlegend([ 'ACC', 'F1'])


def Logistic_Train(self, data, labels, rate, iterations, penaltyL1=0, penaltyL2=0):
    def Gradient(w):
        data1, labels1 = self.Batchify(data, labels, 100)
        exps = np.zeros(len(data1))
        for i in range(len(data1)):
            exps[i] = np.exp(np.dot(w, data1[i]))

        coeffs = 2 * (labels1 - 1 / exps) * (1 / exps)
        grad = np.transpose(np.matmul(coeffs, data1)) + penaltyL1 * np.sign(w) + penaltyL2 * w
        gradmax = max(np.absolute(grad))
        if gradmax:
            return grad / gradmax
        else:
            return grad

    if not self.Logistic_trained:
        self.Logistic_w = np.ones(len(data[0])) / (len(data[0]) ** 7)
        self.Logistic_trained = True
    for i in range(iterations):
        self.Logistic_w += -rate * Gradient(self.Logistic_w)


def Logistic_Return(self, x):
    return np.matmul(np.transpose(self.Logistic_w), x)

for k in range(0, k_max):

    y_predonehot = logisticregression(train_images, train_labels1, test_images, lambdap=100*math.exp(-k))
    test_pred=_pred = np.argmax(y_predonehot, axis=1)
    x_graph[k] = k
    F1scores[k] =metrics.f1_score(test_labels,test_pred,average='weighted')
    ACCscores[k] =metrics.accuracy_score(test_labels,test_pred)
    print(k,metrics.accuracy_score(test_labels,test_pred))
plotter(x_graph,AUCscores,F1scores,"Linear CLassifier with L2 regulariser","Regulariser ( $\lambda $)=100*math.exp(-k) ")
plt.legend([ 'Accuracy','Accuracy']);
plt.ylabel('Accuracy')
plt.show()
