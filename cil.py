#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *

class DifferentErrorCost():
    def __init__(self, kernel=None, beta=1., magic=1.):
        self.kernel = kernel if kernel is not None else GaussKernel(beta)
        self.magic = magic

    def fit(self, posData, negData, label=[1,-1]):
        # equip weight of each class
        numPos, numNeg = float(len(posData)), float(len(negData))
        if numPos < numNeg:
            cPos, cNeg = 1., numPos / numNeg
        else:
            cPos, cNeg = numNeg / numPos, 1.

        # equip sample with these label
        posData = np.c_[ [label[0]]*posData.shape[0], posData]
        negData = np.c_[ [label[1]]*negData.shape[0], negData]

        # concatenate sample matrices
        self.sample = np.r_[posData, negData]

        # ready and fit SVM to given sample
        self.clf = svm.SVC(kernel='precomputed', class_weight={label[0]:cPos, label[1]:cNeg})
        gram = self.kernel.gram(self.sample[:,1:])
        self.clf.fit(gram, self.sample[:,0])

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample[:,1:])
        return self.clf.predict(mat)

class ProbFuzzySVM():
    def __init__(self, beta=1., magic=1., boost=False):
        self.beta = beta
        self.magic = magic
        self.boost = boost

    def fit(self, posData, negData, label=[1,-1]):
        # equip weights based on KDE
        kde = KernelDensityEstimater(self.beta)

        kde.fit(posData)
        posProb = kde.estimate(posData)
        kde.fit(negData)
        negProb = kde.estimate(negData)

        # equip weight og each class
        cPos, cNeg = 1., 1.
        if self.boost is True:
            numPos, numNeg = float(len(posData)), float(len(negData))
            if numPos < numNeg:
                cPos, cNeg = 1., numPos / numNeg
            else:
                cPos, cNeg = numNeg / numPos, 1.

        # create weights array, putting positive's one before negative' one
        self.weights = self.magic * np.r_[posProb, negProb]

        # equip sample with these label
        posData = np.c_[ [label[0]]*posData.shape[0], posData]
        negData = np.c_[ [label[1]]*negData.shape[0], negData]

        # concatenate sample matrices, putting positive's one before negative' one
        self.sample = np.r_[posData, negData]

        # ready and fit SVM to given sample
        self.clf = svm.SVC(kernel='precomputed', class_weight={label[0]:cPos, label[1]:cNeg})
        self.kernel = GaussKernel(self.beta)
        gram = self.kernel.gram(self.sample[:,1:])
        self.clf.fit(gram, self.sample[:,0], sample_weight=self.weights)

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample[:,1:])
        return self.clf.predict(mat)

def dataset_iterator(dataset, nCV, label_index=0, label=[1,-1], shuffle=False):
    pDataset = dataset[dataset[:,label_index]==label[0]]
    pw = len(pDataset) / nCV
    nDataset = dataset[dataset[:,label_index]==label[1]]
    nw = len(nDataset) / nCV

    for i in range(nCV):
        pPiv, nPiv = i*pw, i*nw

        # slice out X(Y) from pos/neg dataset
        if i < nCV -1:
            pX = pDataset[pPiv:pPiv+pw]
            nX = nDataset[nPiv:nPiv+nw]
            pY = np.r_[pDataset[:pPiv],pDataset[pPiv+pw:]]
            nY = np.r_[nDataset[:nPiv],nDataset[nPiv+nw:]]
            X, Y = np.r_[pX,nX], np.r_[pY, nY]
        else:
            X = np.r_[pDataset[pPiv:], nDataset[nPiv:]]
            Y = np.r_[pDataset[:pPiv], nDataset[:nPiv]]

        # if given shuffle flag
        if shuffle is True:
            np.random.shuffle(X)
            np.random.shuffle(Y)

        # slice out label(answer) from X(Y)
        lbl, ans = X[:,label_index], Y[:,label_index]

        # slice out train(test)data from X(Y)
        if label_index >= 0:
            traindata = np.c_[X[:,:label_index:], X[:,label_index+1:]]
            testdata = np.c_[Y[:,:label_index:], Y[:,label_index+1:]]
        else:
            # if given label index is negative,
            # forcibly use -1 as index number
            traindata = X[:,:-1]
            testdata = Y[:,:-1]

        yield (traindata,lbl,testdata,ans)

def procedure(N, classRatio, nCV):
    mean, cov = [-10, -10], [[50,0],[0,100]]
    posDist = NormalDistribution(mean, cov)

    mean, cov = [10, 10], [[100,0],[0,50]]
    negDist = NormalDistribution(mean, cov)

    id = ImbalancedData(posDist, negDist, classRatio)
    dataset = id.getSample(N)

    scores = { "SVM":[], "DEC":[], "PFSVM":[], "PFSVMCIL":[] }
    for X,label,Y,answer in dataset_iterator(dataset, nCV):
        print "given positive samples (train): ", len(label[label[:]==1])
        print "given negative samples (train): ", len(label[label[:]==-1])
        print "given positive samples (test): ", len(answer[answer[:]==1])
        print "given negative samples (test): ", len(answer[answer[:]==-1])

        # params definition
        magic = 20000
        beta = 0.005

        # default SVM
        clf = svm.SVC(kernel='rbf', gamma=beta)
        clf.fit(X, label)
        predict = clf.predict(Y)
        scores['SVM'].append( evaluation(predict, answer) )

        # DEC
        dec = DifferentErrorCost(beta=beta, magic=magic)
        dec.fit(X[label[:]==1,:], X[label[:]==-1,:])
        predict = dec.predict(Y)
        scores['DEC'].append( evaluation(predict, answer) )

        # ProvFuzzySVM
        pfsvm = ProbFuzzySVM(beta=beta, magic=magic)
        pfsvm.fit(X[label[:]==1,:], X[label[:]==-1,:])
        predict = pfsvm.predict(Y)
        scores['PFSVM'].append( evaluation(predict, answer) )

        # ProvFuzzySVM-CIL
        pfsvmcil = ProbFuzzySVM(beta=beta, magic=magic, boost=True)
        pfsvmcil.fit(X[label[:]==1,:], X[label[:]==-1,:])
        predict = pfsvmcil.predict(Y)
        scores['PFSVMCIL'].append( evaluation(predict, answer) )

    for k, v in scores.items():
        tmp = np.array(v)
        print "%s:\t acc:\t %s" % (k, sum(tmp[:,0]) / nCV)
        print "%s:\t accP:\t %s" % (k, sum(tmp[:,1]) / nCV)
        print "%s:\t accN:\t %s" % (k, sum(tmp[:,2]) / nCV)
        print "%s:\t g:\t %s" % (k, sum(tmp[:,3]) / nCV)

if __name__ == '__main__':
    for cr in [1., 2., 5., 10., 20., 50., 100.]:
        #procedure(5000, cr, 5)
        procedure(1234, cr, 3)
