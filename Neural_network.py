import dill
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from scipy.special import expit


def ReLU(x):
    return np.where(x>0,x,0.01*x)


def ReLU_derivative(x):
    return np.where(x>0,1,0.01)


def sigmoid(x):
    return expit(x)


def sigmoid_derivative(x):
    f = sigmoid(x)
    return f * (1 - f)


def MSE(y, y_hat):
    return np.mean((y - y_hat) ** 2)


class Neural_net():
    def __init__(self, input_n, hidden_n, output_n, *, activation_function=ReLU,
                 activation_function_derivative=ReLU_derivative):
        # assummption that hidden layer is a list of hidden layer sizes
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.make_layers(activation_function, activation_function_derivative)

    def make_layers(self, activation_function, activation_function_derivative):
        self.input_layer = np.zeros(self.input_n)
        self.hidden_layer = []
        if type(activation_function) != list:
            activation_function = [activation_function] * (len(self.hidden_n) + 1)
            activation_function[-1] = sigmoid
            activation_function_derivative = [activation_function_derivative] * (len(self.hidden_n) + 1)
            activation_function_derivative[-1] = sigmoid_derivative
        else:
            assert len(activation_function) == len(self.hidden_n) + 1
            assert len(activation_function_derivative) == len(self.hidden_n) + 1
        inp_dim = self.input_n
        for i in range(len(self.hidden_n)):
            self.hidden_layer.append(
                Layer(inp_dim, self.hidden_n[i], activation_function[i], activation_function_derivative[i]))
            inp_dim = self.hidden_n[i]

        self.output_layer = Layer(inp_dim, self.output_n, activation_function[-1],
                                  activation_function_derivative[-1])

    def forward_prop(self, input_data):
        self.input_layer = input_data
        for i in range(len(self.hidden_layer)):
            self.hidden_layer[i].forward_prop(input_data)
            input_data = self.hidden_layer[i].z
        self.output_layer.forward_prop(input_data)
        return self.output_layer.z

    def back_prop(self, target, learning_rate=0.01):
        self.output_layer.back_prop(self.delta_last(target), learning_rate)
        delta_s = self.delta_sum(self.output_layer)
        for i in range(len(self.hidden_layer) - 1, -1, -1):
            layer = self.hidden_layer[i]
            delta = delta_s * layer.activation_function_derivative(layer.a)
            self.hidden_layer[i].back_prop(delta, learning_rate)
            delta_s = self.delta_sum(self.hidden_layer[i])
        return self.output_layer.delta

    def delta_sum(self, layer):
        delta_thing = np.zeros(layer.weights.shape[1])
        for j in range(delta_thing.shape[0]):
            delta_thing[j] = np.sum(layer.weights[:, j] * layer.delta)
        return delta_thing

    def delta_last(self, target):
        return (self.output_layer.z - target) * self.output_layer.activation_function_derivative(self.output_layer.z)


class Layer():
    def __init__(self, input_dim, output_dim, activation_function, activation_function_derivative):
        glorat = np.sqrt(2 / (input_dim + output_dim))
        self.weights = np.random.randn(output_dim, input_dim) * glorat
        self.bias = np.random.randn(output_dim) * glorat
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.a = np.zeros(output_dim)
        self.z = activation_function(self.a)
        self.prevz = np.zeros(input_dim)
        self.l1=0
        self.l2=0
    def forward_prop(self, input_data):
        self.prevz = input_data
        self.a = np.dot(self.weights, input_data) + self.bias
        self.z = self.activation_function(self.a)

    def back_prop(self, delta, learning_rate):

        self.delta = delta
        a = self.delta.reshape(self.delta.shape[0], 1)
        b= self.prevz.reshape(1,self.prevz.shape[0])
        self.weights_gradient = np.matmul(a,b)+self.l2*self.weights+self.l1*np.sign(self.weights)
        assert (self.weights_gradient.shape == self.weights.shape)
        self.bias_gradient = self.delta+self.l2*self.bias+self.l1*np.sign(self.bias)
        self.update_weights(learning_rate)

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient


def one_hot_encoding(labels):
    one_hot_labels = np.zeros((labels.shape[0], 2))
    for i in range(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels


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


def normalize_images(images: np):
    new_images = np.zeros((images.shape[0], images.shape[1] * images.shape[2]))
    for i in range(images.shape[0]):
        new_images[i] = images[i].flatten()
    new_images = new_images.astype(np.float32)
    new_images /= 255.0
    return new_images


if __name__ == '__main__':
    path = "pneumoniamnist.npz"
    train_images, val_images, test_images, train_labels1, val_labels1, test_labels1 = load_input(path)
    train_labels = one_hot_encoding(train_labels1)
    val_labels = one_hot_encoding(val_labels1)
    test_labels = one_hot_encoding(test_labels1)
    inp_size = train_images.shape[1]
    out_size = train_labels.shape[1]
    nn = Neural_net(inp_size, [100,100], out_size)
    nn.hidden_layer[0].l2=0.001
    nn.hidden_layer[1].l1 =0.001
    nn.hidden_layer[0].l2 = 0.0001
    loss=[]
    pred = np.zeros(train_labels.shape)
    for i in range(1):
        for j in range(train_images.shape[0]):
            l = nn.forward_prop(train_images[j])
            nn.back_prop(train_labels[j],0.005)
            if (j % 100 == 0):
                #print(l)
                for k in range(train_images.shape[0]):
                    pred[k] = nn.forward_prop(train_images[k])
                loss.append( MSE(pred, train_labels))
                print(loss)

    with open('neuralnet.txt', 'wb') as f:
        dill.dump(nn,f)
    plt.plot(loss)
    plt.xlabel("Iterations*100")
    plt.ylabel("Loss")

    plt.show()

     #with open('neuralnet.txt', 'rb') as f:
        #nn = dill.load(f)


    pred_test = np.zeros(test_labels1.shape)
    for j in range(test_images.shape[0]):
        pred_test[j] = np.argmax(nn.forward_prop(test_images[j]))
    acc = metrics.accuracy_score(pred_test, test_labels1)
    print(acc)
    auc=metrics.roc_auc_score(test_labels1,pred_test)
    print(auc)
    f1= metrics.f1_score(pred_test, test_labels1)
    print(f1)

    """a = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    b = np.array([[0.0], [1], [1], [0.0]])
    nn = Neural_net(2, [3, 2], 1)

    for i in range(100000):
        for j in range(4):
            nn.forward_prop(a[j])
            nn.back_prop(b[j], 0.1)
        if (i % 1000 == 0):
            print(nn.forward_prop(a[0]), nn.forward_prop(a[1]), nn.forward_prop(a[2]), nn.forward_prop(a[3]), sep=" ,")"""
