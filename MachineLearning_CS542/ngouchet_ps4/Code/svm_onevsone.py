import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import scipy.io
import numpy as np
import itertools
from pandas_ml import ConfusionMatrix

mat = scipy.io.loadmat('MNIST_data.mat')
X_train = np.array(mat['train_samples'])
y_train = np.array(mat['train_samples_labels']).reshape((mat['train_samples_labels'].shape[0],))
X_test = np.array(mat['test_samples'])
y_test = np.array(mat['test_samples_labels']).reshape((mat['test_samples_labels'].shape[0],))

def data_clustering(X_train, y_train):
    X_train0 = []
    X_train1 = []
    X_train2 = []
    X_train3 = []
    X_train4 = []
    X_train5 = []
    X_train6 = []
    X_train7 = []
    X_train8 = []
    X_train9 = []

    for i in xrange(X_train.shape[0]):
        if y_train[i] == 0:
            X_train0.append(X_train[i])
        elif y_train[i] == 1:
            X_train1.append(X_train[i])
        elif y_train[i] == 2:
            X_train2.append(X_train[i])
        elif y_train[i] == 3:
            X_train3.append(X_train[i])
        elif y_train[i] == 4:
            X_train4.append(X_train[i])
        elif y_train[i] == 5:
            X_train5.append(X_train[i])
        elif y_train[i] == 6:
            X_train6.append(X_train[i])
        elif y_train[i] == 7:
            X_train7.append(X_train[i])
        elif y_train[i] == 8:
            X_train8.append(X_train[i])
        elif y_train[i] == 9:
            X_train9.append(X_train[i])

    return np.array(X_train0), np.array(X_train1), np.array(X_train2), np.array(X_train3), np.array(X_train4), np.array(X_train5), np.array(X_train6), np.array(X_train7), np.array(X_train8), np.array(X_train9)


def generate_data(X1, X2):
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1
    return y1 , y2


def transform_data(predict, plus, minus):
    for i in xrange(predict.shape[0]):
        if predict[i] == 1 :
            predict[i] = plus
        elif predict[i] == -1 :
            predict[i] = minus

    return predict


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=polynomial_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def train(self, X, y):
        n_samples, n_features = X.shape
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # Intercept
        self.b = 0
        
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def compute(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.compute(X))

def one_vs_one():
    X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9 = data_clustering(X_train, y_train)
    prediction = []
    numpy_list = []
    numpy_predict = [X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9]
    combination  = list(itertools.combinations([0,1,2,3,4,5,6,7,8,9], 2))

    for pair in combination:
        y1, y2 = generate_data(numpy_predict[pair[0]], numpy_predict[pair[1]])
        training_data = np.vstack((numpy_predict[pair[0]] , numpy_predict[pair[1]]))
        test_data = np.hstack((y1, y2))
        clf = SVM(C=0.1)
        clf.train(training_data, test_data)
        y_predict = clf.predict(X_test)
        numpy_list.append(transform_data(y_predict, pair[0], pair[1]))

    numpy_list = np.array(numpy_list).astype(int)
    transpose = np.transpose(numpy_list)

    for row in xrange(transpose.shape[0]):
        counts = np.bincount(transpose[row])
        prediction.append(np.argmax(counts))


    prediction = np.array(prediction)
    correct = np.sum(prediction == y_test)
    confusion_matrix = ConfusionMatrix(y_test, prediction)
    print("Confusion matrix:\n%s" % confusion_matrix)
    size = len(y_predict)
    accuracy = (correct/float(size)) * 100
    print "%d out of %d predictions correct" % (correct, len(y_predict))
    print "The accuracy in percentage is  "
    print(accuracy)

    
one_vs_one()
