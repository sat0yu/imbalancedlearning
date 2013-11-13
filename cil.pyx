#coding: utf-8;
import numpy as np
cimport numpy as np

from sklearn import svm

from kernel import *
from mlutil import *

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t
np.seterr(over='raise')

class DifferentErrorCosts(FloatKernel):
    def __init__(self, kernel):
        self.kernel = kernel

    def class_weight(self, label, class_label=[1,-1]):
        numPos = float(len(label[label[:]==class_label[0]]))
        numNeg = float(len(label[label[:]==class_label[1]]))

        if numPos < numNeg:
            cPos, cNeg = 1., numPos / numNeg
        else:
            cPos, cNeg = numNeg / numPos, 1.

        return (cPos, cNeg)

    def precompute(self, sample):
        return self.kernel.gram(sample)

    def fit(self, sample, label, C=1., class_label=[1,-1], gram=None):
        # equip weight of each class
        cPos, cNeg = self.class_weight(label, class_label)

        self.sample = sample
        # NOT precomputed
        if gram is None:
            gram = self.precompute(sample)

        # ready and fit SVM
        self.clf = svm.SVC(kernel='precomputed', C=C, class_weight={label[0]:cPos, label[1]:cNeg})
        self.clf.fit(gram, label)

    def predict(self, target, precomputed=False):
        if precomputed is False:
            target = self.kernel.matrix(target, self.sample)

        return self.clf.predict(target)

class KernelProbabilityFuzzySVM(FloatKernel):
    def __init__(self, kernel):
        self.kernel = kernel

    def class_weight(self, sample, label, class_label=[1,-1]):
        numPos = float(len(label[label[:]==class_label[0]]))
        numNeg = float(len(label[label[:]==class_label[1]]))

        if numPos < numNeg:
            cPos, cNeg = 1., numPos / numNeg
        else:
            cPos, cNeg = numNeg / numPos, 1.

        return (cPos, cNeg)

    def precompute(self, sample, label, class_label=[1,-1]):
        # sort given sample by their label
        dataset = np.c_[label, sample]
        dataset = dataset[dataset[:,0].argsort()]
        label, sample = dataset[:,0], dataset[:,1:]

        # count sample belong to each class
        numPos = len(label[label[:]==class_label[0]])
        numNeg = len(label[label[:]==class_label[1]])

        # calc. gram matrix and then sample_weight
        gram = self.kernel.gram(sample)
        nFront, nBack = (numNeg, numPos) if class_label[0] > class_label[1] else (numPos, numNeg)
        wFront = np.sum(gram[:nFront,:nFront], axis=0)
        wBack = np.sum(gram[nFront:,nFront:], axis=0)
        weight = np.r_[wFront / nFront, wBack / nBack]

        return (sample, gram, label, weight)

    def fit(self, sample, label, C=1., class_label=[1,-1], gram=None, sample_weight=None):
        # equip weight of each class
        cPos, cNeg = self.class_weight(sample, label, class_label)

        # given both gram matrix and sample_weight
        if gram is not None and sample_weight is not None:
            self.sample = sample
        # NOT precomputed
        else:
            self.sample, gram, label, sample_weight = self.precompute(sample, label, class_label)

        # ready and fit SVM
        self.clf = svm.SVC(kernel='precomputed', C=C, class_weight={label[0]:cPos, label[1]:cNeg})
        self.clf.fit(gram, label, sample_weight=sample_weight)

    def predict(self, target, precomputed=False):
        if precomputed is False:
            target = self.kernel.matrix(target, self.sample)

        return self.clf.predict(target)

class FSVMCIL():
    def __init__(self, beta, distance_function='center', decay_function='linear', delta=1., gamma=1.):
        self.beta = beta
        self.delta = delta
        self.gamma = gamma

        if distance_function == 'center':
            self.distance_function = self.dist_from_center
        elif distance_function == 'estimate':
            self.distance_function = self.dist_from_estimated_hyperplane
        elif distance_function == 'hyperplane':
            self.distance_function = self.dist_from_hyperplane
        else:
            raise ValueError("the argument named dist_from expects a velue of 'center' or 'estimate', 'hyperplane'")

        if decay_function == 'linear':
            self.decay_function = self.linear_decay_function
        elif decay_function == 'exp':
            self.decay_function = self.exp_decay_function
        else:
            raise ValueError("the argument named decay_function expects a velue of 'linear' or 'exp'")

    def linear_decay_function(self, X):
        denominator = max(X) + self.delta
        return 1. - (X / denominator)

    def exp_decay_function(self, np.ndarray[DTYPE_float_t, ndim=1] X):
        cdef int i
        cdef np.ndarray[DTYPE_float_t, ndim=1] ret = np.zeros_like(X)

        for i in range(len(X)):
            try:
                ret[i] = 2. / ( 1. + np.exp(self.gamma * X[i]) )

            except FloatingPointError:
                ret[i] = 10**(-323)

        return ret

    def class_weight(self, label, class_label=[1,-1]):
        numPos = float(len(label[label[:]==class_label[0]]))
        numNeg = float(len(label[label[:]==class_label[1]]))

        if numPos < numNeg:
            cPos, cNeg = 1., numPos / numNeg
        else:
            cPos, cNeg = numNeg / numPos, 1.

        return (cPos, cNeg)

    def dist_from_center(self, X, label, C=1., class_label=[1,-1]):
        def dist(_X):
            # calc. center
            center = np.average(_X, axis=0)
            # calc. distance between from center for each sample
            return np.sum(np.abs(_X - center)**2, axis=-1)**(1/2.)

        # sort given sample with their label
        dataset = np.c_[label, X]
        dataset = dataset[dataset[:,0].argsort()]
        label, X = dataset[:,0], dataset[:,1:]

        # separate given samples according to theirs label
        numNeg = len(label[label[:]==class_label[1]])
        negData, posData  = X[:int(numNeg)], X[int(numNeg):]

        # concatenate arrays
        distance = np.r_[dist(negData), dist(posData)]

        return (X, label, distance)

    def dist_from_estimated_hyperplane(self, X, label, C=1., class_label=[1,-1]):
        # sort given sample with their label
        dataset = np.c_[label, X]
        dataset = dataset[dataset[:,0].argsort()]
        label, X = dataset[:,0], dataset[:,1:]

        # calc. gram matrix and then sample_weight
        kernel = GaussKernel(self.beta)
        gram = kernel.gram(X)
        weight = np.dot(np.diag(label), np.dot(gram, label))

        return (X, label, weight)

    def dist_from_hyperplane(self, X, label, C=1., class_label=[1,-1]):
        # train conventional SVM
        clf = svm.SVC(kernel='rbf', gamma=self.beta, C=C)
        clf.fit(X, label)

        # calc. distance between from hyperplane
        value = (clf.decision_function(X))[:,0]
        distance = np.abs(value)

        return (X, label, distance)

    def fit(self, sample, label, C=1., class_label=[1,-1], gram=None, weight=None):
        # equip weight of each class
        cPos, cNeg = self.class_weight(label, class_label)

        if gram is not None and weight is not None:
            self.clf = svm.SVC(kernel='precomputed', C=C, class_weight={label[0]:cPos, label[1]:cNeg})

            self.clf.fit(gram, label, sample_weight=weight)

        else:
            # calc. dist
            sample, label, dist = self.distance_function(sample, label, C=C, class_label=class_label)

            # apply decay function
            weight = self.decay_function(dist)

            # ready and fit SVM to given sample
            self.clf = svm.SVC(kernel='rbf', gamma=self.beta, C=C, class_weight={label[0]:cPos, label[1]:cNeg})

            self.clf.fit(sample, label, sample_weight=weight)

    def predict(self, target):
        return self.clf.predict(target)
