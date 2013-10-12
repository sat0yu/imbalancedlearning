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
    rough_C, beta, Y, answer, X, label = args

    clf = KernelProbabilityFuzzySVM( GaussKernel(beta) )
    X, gram, label, weight = clf.precompute(X, label)

    res = []
    for _C in rough_C:
        clf.fit(X, label, C=_C, gram=gram, sample_weight=weight)
        predict = clf.predict(Y)
        res.append( (_C,)+evaluation(predict, answer) )

    return res

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
        for beta in rough_beta:
            args = [ (rough_C, beta) + elem for elem in dataset_iterator(pseudo, nCV) ]
            res = pool.map(multiproc, args)

            res_foreach_dataset = np.array(res)
            #print res_foreach_dataset.shape
            res_foreach_C = np.average(res_foreach_dataset, axis=0)
            #print res_foreach_C.shape

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta

        print "[rough search] opt_beta:%f,\topt_C:%f,\tg:%f" % (opt_beta,opt_C,max_g)
        sys.stdout.flush()

        # narrow parameter search
        max_g = -999.
        narrow_C = [opt_C*(10**j) for j in narrow_space]
        for beta in [opt_beta*(10**i) for i in narrow_space]:
            args = [ (narrow_C, beta) + elem for elem in dataset_iterator(pseudo, nCV) ]
            res = pool.map(multiproc, args)

            res_foreach_dataset = np.array(res)
            #print res_foreach_dataset.shape
            res_foreach_C = np.average(res_foreach_dataset, axis=0)
            #print res_foreach_C.shape

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta

        print "[narrow search] opt_beta:%f,\topt_C:%f,\tg:%f" % (opt_beta,opt_C,max_g)
        sys.stdout.flush()

        # classify using searched params
        gk = GaussKernel(opt_beta)
        clf = KernelProbabilityFuzzySVM(gk)
        clf.fit(X, label, C=opt_C)
        predict = clf.predict(Y)
        scores.append( evaluation(predict, answer) )

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (acc,accP,accN,g,_g)

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
