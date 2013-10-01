#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm
from dataset import *

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *

class DifferentErrorCost():
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, sample, label, class_label=[1,-1]):
        # store given sample
        self.sample = sample

        # equip weight of each class
        numPos = float(len(label[label[:]==class_label[0]]))
        numNeg = float(len(label[label[:]==class_label[1]]))
        if numPos < numNeg:
            cPos, cNeg = 1., numPos / numNeg
        else:
            cPos, cNeg = numNeg / numPos, 1.

        # ready and fit SVM to given sample
        self.clf = svm.SVC(kernel='precomputed', class_weight={class_label[0]:cPos, class_label[1]:cNeg})
        gram = self.kernel.gram(self.sample)
        self.clf.fit(gram, label)

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample)
        return self.clf.predict(mat)

class KernelProbabilityFuzzySVM():
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, sample, label, class_label=[1,-1]):
        # equip weight of each class
        numPos = float(len(label[label[:]==class_label[0]]))
        numNeg = float(len(label[label[:]==class_label[1]]))
        if numPos < numNeg:
            cPos, cNeg = 1., numPos / numNeg
        else:
            cPos, cNeg = numNeg / numPos, 1.

        # sort given sample with their label
        dataset = np.c_[label, sample]
        dataset = dataset[dataset[:,0].argsort()]
        label, self.sample = dataset[:,0], dataset[:,1:]

        # calc. gram matrix and then sample_weight
        gram = self.kernel.gram(self.sample)
        nFront, nBack = (numNeg, numPos) if class_label[0] > class_label[1] else (numPos, numNeg)
        boundary = int(nFront)
        wFront = np.sum(gram[:boundary,:boundary], axis=0)
        wBack = np.sum(gram[boundary:,boundary:], axis=0)
        weight = np.r_[wFront / nFront, wBack / nBack]

        # ready and fit SVM to given sample
        self.clf = svm.SVC(kernel='precomputed', class_weight={label[0]:cPos, label[1]:cNeg})
        self.clf.fit(gram, label, sample_weight=weight)

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample)
        return self.clf.predict(mat)

def procedure(dataset, nCV, **kwargs):
    scores = { "SVM":[], "DEC":[], "KPFSVM":[] }
    for Y,answer,X,label in dataset_iterator(dataset, nCV, **kwargs):
        print "given positive samples (train): ", len(label[label[:]==1])
        print "given negative samples (train): ", len(label[label[:]==-1])
        print "given positive samples (test): ", len(answer[answer[:]==1])
        print "given negative samples (test): ", len(answer[answer[:]==-1])

        # params definition
        beta = 0.005

        # create Kernel Instance(s)
        gk = GaussKernel(beta)

        # default SVM
        clf = svm.SVC(kernel='rbf', gamma=beta)
        clf.fit(X, label)
        predict = clf.predict(Y)
        scores['SVM'].append( evaluation(predict, answer) )

        # DEC
        dec = DifferentErrorCost(gk)
        dec.fit(X, label)
        predict = dec.predict(Y)
        scores['DEC'].append( evaluation(predict, answer) )

        # KernelProbabilityFuzzySVM
        kpfsvm = KernelProbabilityFuzzySVM(gk)
        kpfsvm.fit(X, label)
        predict = kpfsvm.predict(Y)
        scores['KPFSVM'].append( evaluation(predict, answer) )

    for k, v in scores.items():
        tmp = np.array(v)
        print "%s:\t acc:\t %s" % (k, sum(tmp[:,0]) / nCV)
        accP, accN = sum(tmp[:,1])/nCV, sum(tmp[:,2])/nCV
        g = np.sqrt(accP * accN)
        print "%s:\t accP:\t %s" % (k, accP)
        print "%s:\t accN:\t %s" % (k, accN)
        print "%s:\t g:\t %s" % (k, g)

if __name__ == '__main__':
    posDist = NormalDistribution([-10, -10], [[50,0],[0,100]])
    negDist = NormalDistribution([10, 10], [[100,0],[0,50]])
    id = ImbalancedData(posDist, negDist, 50.)
    dataset = id.getSample(5000)
    procedure(dataset, 5, label_index=0)

    page = Dataset("data/page-blocks.rplcd", label_index=-1, dtype=np.float)
    procedure(page.raw, 5, label_index=-1)

    yeast = Dataset("data/yeast.rplcd", label_index=-1, usecols=range(1,10), dtype=np.float)
    procedure(yeast.raw, 5, label_index=-1)
