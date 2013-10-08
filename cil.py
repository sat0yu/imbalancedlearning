#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm
from dataset import *
import multiprocessing

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *

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

def multiproc(args):
    beta, C, Y, answer, X, label = args

    clf = KernelProbabilityFuzzySVM(GaussKernel(beta))
    clf.fit(X, label, C=C)
    predict = clf.predict(Y)

    #acc,accP,accN,g = evaluation(predict, answer)
    #print "[C:%f\tbeta:%f]" % (C,beta)
    #print "%f\t%f\t%f\t%f" % (acc,accP,accN,g)
    #return (acc,accP,accN,g)

    return evaluation(predict, answer)

def procedure(dataset, nCV=5, **kwargs):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    rough_beta = [10**i for i in range(-9,1)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)

    # cross varidation
    scores = []
    for Y,answer,X,label in dataset_iterator(dataset, nCV, **kwargs):
        print "train samples (pos:%d, neg:%d)" % (len(label[label[:]==1]),len(label[label[:]==-1]))
        print "test samples (pos:%d, neg:%d)" % (len(answer[answer[:]==1]),len(answer[answer[:]==-1]))

        # ready parametersearch
        pseudo = np.c_[label, X]
        pool = multiprocessing.Pool()
        opt_beta, opt_C, max_g = 0., 0., -999.

        # rough parameter search
        for C in rough_C:
            for beta in rough_beta:
                args = [ (beta, C) + elem for elem in dataset_iterator(pseudo, nCV) ]
                buf = pool.map(multiproc, args)

                acc, accP, accN = np.average(np.array(buf), axis=0)[:3]
                g = np.sqrt(accP * accN)
                if g > max_g:
                    max_g, opt_C, opt_beta = g, C, beta
        print "[rough search] opt_beta:%f,\topt_C:%f,\tg:%f" % (opt_beta,opt_C,max_g)

        # narrow parameter search
        max_g = -999.
        for C in [opt_C*(10**j) for j in narrow_space]:
            for beta in [opt_beta*(10**i) for i in narrow_space]:
                args = [ (beta, C) + elem for elem in dataset_iterator(pseudo, nCV) ]
                buf = pool.map(multiproc, args)

                acc, accP, accN = np.average(np.array(buf), axis=0)[:3]
                g = np.sqrt(accP * accN)
                if g > max_g:
                    max_g, opt_C, opt_beta = g, C, beta
        print "[narrow search] opt_beta:%f,\topt_C:%f,\tg:%f" % (opt_beta,opt_C,max_g)

        # classify using searched params
        gk = GaussKernel(opt_beta)
        clf = KernelProbabilityFuzzySVM(gk)
        clf.fit(X, label)
        predict = clf.predict(Y)
        scores.append( evaluation(predict, answer) )

    # average evaluation score
    acc, accP, accN = np.average(np.array(scores), axis=0)[:3]
    g = np.sqrt(accP * accN)
    print "acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % (acc,accP,accN,g)

if __name__ == '__main__':
    posDist = NormalDistribution([-10, -10], [[50,0],[0,100]])
    negDist = NormalDistribution([10, 10], [[100,0],[0,50]])
    id = ImbalancedData(posDist, negDist, 50.)
    dataset = id.getSample(5000)
    procedure(dataset, nCV=4, label_index=0)

    page = Dataset("data/page-blocks.rplcd", label_index=-1, dtype=np.float)
    procedure(page.raw, label_index=-1)

    yeast = Dataset("data/yeast.rplcd", label_index=-1, usecols=range(1,10), dtype=np.float)
    procedure(yeast.raw, label_index=-1)

    abalone = Dataset("data/abalone.rplcd", label_index=-1, usecols=range(1,9), delimiter=',', dtype=np.float)
    procedure(abalone.raw, label_index=-1)

    ecoli = Dataset("data/ecoli.rplcd", label_index=-1, usecols=range(1,9), dtype=np.float)
    procedure(ecoli.raw, label_index=-1)

    transfusion = Dataset("data/transfusion.rplcd", label_index=-1, delimiter=',', skiprows=1, dtype=np.float)
    procedure(transfusion.raw, label_index=-1)

    haberman = Dataset("data/haberman.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    procedure(haberman.raw, label_index=-1)

    waveform = Dataset("data/waveform.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    procedure(waveform.raw, label_index=-1)

    pima = Dataset("data/pima-indians-diabetes.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    procedure(pima.raw, label_index=-1)
