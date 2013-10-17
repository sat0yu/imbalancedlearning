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
from cil import *

def dataset_iterator(data, label, nCV, label_value=[1,-1]):
    pData = data[label[:]==label_value[0]]
    pw = len(pData) / nCV
    nData = data[label[:]==label_value[1]]
    nw = len(nData) / nCV

    for i in range(nCV):
        pPiv, nPiv = i*pw, i*nw

        # slice out X(Y)/label(answer) from pos/neg data
        if i < nCV -1:
            pX = pData[pPiv:pPiv+pw]
            nX = nData[nPiv:nPiv+nw]
            pY = np.r_[pData[:pPiv],pData[pPiv+pw:]]
            nY = np.r_[nData[:nPiv],nData[nPiv+nw:]]
            # PARTICULAR PROCESS OF NON-VECTORIAL(in this case, string) DATA
            # PLACE NEGATIVE DATA BEFORE POSITIVES
            X, Y = np.r_[nX,pX], np.r_[nY, pY]
            label = np.r_[ [label_value[1]]*len(nX), [label_value[0]]*len(pX) ]
            answer = np.r_[ [label_value[1]]*len(nY), [label_value[0]]*len(pY) ]
        else:
            pX, nX = pData[pPiv:], nData[nPiv:]
            pY, nY= pData[:pPiv], nData[:nPiv]
            # PARTICULAR PROCESS OF NON-VECTORIAL(in this case, string) DATA
            # PLACE NEGATIVE DATA BEFORE POSITIVES
            X, Y = np.r_[nX,pX], np.r_[nY, pY]
            label = np.r_[ [label_value[1]]*len(nX), [label_value[0]]*len(pX) ]
            answer = np.r_[ [label_value[1]]*len(nY), [label_value[0]]*len(pY) ]

        yield (X, label, Y, answer)

def multiproc(args):
    rough_C, beta, Y, answer, X, label = args

    clf = KernelProbabilityFuzzySVM( GaussKernel(beta) )
    #clf = DifferentErrorCosts( GaussKernel(beta) )
    X, gram, label, weight = clf.precompute(X, label)
    #gram = clf.precompute(X)

    res = []
    for _C in rough_C:
        clf.fit(X, label, C=_C, gram=gram, sample_weight=weight)
        #clf.fit(X, label, C=_C, gram=gram)
        predict = clf.predict(Y)
        res.append( (_C,)+evaluation(predict, answer) )

    return res

def procedure(X, label, p_list, nCV=5):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)

    # cross varidation
    scores = []
    for i_CV, (Y,answer,X,label) in enumerate( dataset_iterator(X, label, nCV) ):
        pos, neg = len(label[label[:]==1]),len(label[label[:]==-1])
        print "[%d/%d]: train samples (pos:%d, neg:%d)" % (i_CV, nCV, pos, neg)
        pos, neg = len(answer[answer[:]==1]),len(answer[answer[:]==-1])
        print "[%d/%d]: test samples (pos:%d, neg:%d)" % (i_CV, nCV, pos, neg)

        # ready parametersearch
        pseudo = np.c_[label, X]
        pool = multiprocessing.Pool()
        opt_beta, opt_C, max_g = 0., 0., -999.

        # rough parameter search
        for beta in rough_beta:
            args = [ (rough_C, beta) + elem for elem in dataset_iterator(X, label, nCV) ]
            res = pool.map(multiproc, args)

            res_foreach_dataset = np.array(res)
            #print res_foreach_dataset.shape
            res_foreach_C = np.average(res_foreach_dataset, axis=0)
            #print res_foreach_C.shape

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta

        print "[rough search] opt_beta:%s,\topt_C:%s,\tg:%f" % (opt_beta,opt_C,max_g)
        sys.stdout.flush()

        # narrow parameter search
        max_g = -999.
        narrow_C = [opt_C*(10**j) for j in narrow_space]
        for beta in [opt_beta*(10**i) for i in narrow_space]:
            args = [ (narrow_C, beta) + elem for elem in dataset_iterator(X, label, nCV) ]
            res = pool.map(multiproc, args)

            res_foreach_dataset = np.array(res)
            #print res_foreach_dataset.shape
            res_foreach_C = np.average(res_foreach_dataset, axis=0)
            #print res_foreach_C.shape

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta

        print "[narrow search] opt_beta:%s,\topt_C:%s,\tg:%f" % (opt_beta,opt_C,max_g)
        sys.stdout.flush()

        # classify using searched params
        gk = GaussKernel(opt_beta)
        #clf = DifferentErrorCosts(gk)
        clf = KernelProbabilityFuzzySVM(gk)
        clf.fit(X, label, C=opt_C)
        predict = clf.predict(Y)
        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (acc,accP,accN,g,_g)

if __name__ == '__main__':
    spam = Dataset("data/SMSSpamCollection.rplcd", isNonvectorial=True, delimiter='\t', dtype={'names':('0','1'), 'formats':('f8','S512')})
    label = spam.raw['0']
    X = spam.raw['1']

    procedure(X, label, range(2,11), nCV=5)
