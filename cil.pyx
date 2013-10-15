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

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample)
        return self.clf.predict(mat)

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

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample)
        return self.clf.predict(mat)

