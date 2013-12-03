#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm
from dataset import *
import multiprocessing

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from nonvectorial import *
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

def precompute(kernel, sample, label, target, class_label=[1,-1]):
    # count sample belong to each class
    numPos = len(label[label[:]==class_label[0]])
    numNeg = len(label[label[:]==class_label[1]])

    # calc. gram matrix and then sample_weight
    gram = kernel.gram(sample)
    mat = kernel.matrix(target, sample)
    # NOW, WE PUT THE ASSUMPTION THAT POSITIVE(NEGATIVE) LABEL IS 1(-1)
    # AND GRAM MATRIX IS CREATED FROM THE DATA IN WHICH NEGATIVE DATA IS PLACED BEFORE POSITIVES
    nFront, nBack = numNeg, numPos
    wFront = np.sum(gram[:nFront,:nFront], axis=0)
    wBack = np.sum(gram[nFront:,nFront:], axis=0)
    weight = np.r_[wFront / nFront, wBack / nBack]

    return (gram, mat, weight)

def multiproc(args):
    rough_C, p, Y, answer, X, label = args

    ## <SVM>
    #sk = NormalizedSpectrumKernel(p)
    #gram = sk.gram(X)
    #mat = sk.matrix(Y,X)
    ## </SVM>

    ## <Differenterrorcosts>
    sk = NormalizedSpectrumKernel(p)
    clf = DifferentErrorCosts(sk)
    gram = sk.gram(X)
    mat = sk.matrix(Y,X)
    ## </Differenterrorcosts>

    ## <Kernelprobabilityfuzzysvm>
    #sk = NormalizedSpectrumKernel(p)
    #clf = KernelProbabilityFuzzySVM(sk)
    #gram, mat, weight = precompute(sk, X, label, Y)
    ## </Kernelprobabilityfuzzysvm>

    res = []
    for _C in rough_C:
        ## <SVM>
        #clf = svm.SVC(kernel='precomputed', C=_C)
        #clf.fit(gram, label)
        #predict = clf.predict(mat)
        ## </SVM>

        ## <Differenterrorcosts>
        clf.fit(X, label, C=_C, gram=gram)
        predict = clf.predict(mat, precomputed=True)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        #clf.fit(X, label, C=_C, gram=gram, sample_weight=weight)
        #predict = clf.predict(mat, precomputed=True)
        ## <Kernelprobabilityfuzzysvm>

        res.append( (_C,)+evaluation(predict, answer) )

    return res

def procedure(dataname, stringdata, datalabel, p, nCV=5):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)

    # cross varidation
    scores = []
    for i_CV, (Y,answer,X,label) in enumerate( dataset_iterator(stringdata, datalabel, nCV) ):
        pos, neg = len(label[label[:]==1]),len(label[label[:]==-1])
        print "%s[%d/%d]: train samples (pos:%d, neg:%d)" % (dataname, i_CV, nCV, pos, neg)
        pos, neg = len(answer[answer[:]==1]),len(answer[answer[:]==-1])
        print "%s[%d/%d]: test samples (pos:%d, neg:%d)" % (dataname, i_CV, nCV, pos, neg)

        # ready parametersearch
        pool = multiprocessing.Pool(2)
        opt_C = 0.

        # rough parameter search
        max_g = -999.
        args = [ (rough_C, p) + elem for elem in dataset_iterator(X, label, nCV) ]
        res = pool.map(multiproc, args)

        res_foreach_dataset = np.array(res)
        #print res_foreach_dataset.shape
        res_foreach_C = np.average(res_foreach_dataset, axis=0)
        #print res_foreach_C.shape

        for _C, _acc, _accP, _accN, _g in res_foreach_C:
            _g = np.sqrt(_accP * _accN)
            if _g > max_g: max_g, opt_C = _g, _C

        print "[rough search] opt_C:%s,\tg:%f" % (opt_C,max_g)
        sys.stdout.flush()

        # narrow parameter search
        max_g = -999.
        narrow_C = [opt_C*(10**j) for j in narrow_space]

        args = [ (narrow_C, p) + elem for elem in dataset_iterator(X, label, nCV) ]
        res = pool.map(multiproc, args)

        res_foreach_dataset = np.array(res)
        #print res_foreach_dataset.shape
        res_foreach_C = np.average(res_foreach_dataset, axis=0)
        #print res_foreach_C.shape

        for _C, _acc, _accP, _accN, _g in res_foreach_C:
            _g = np.sqrt(_accP * _accN)
            if _g > max_g: max_g, opt_C = _g, _C

        print "[narrow search] opt_C:%s,\tg:%f" % (opt_C,max_g)
        sys.stdout.flush()

        # classify using searched params

        ## <SVM>
        #sk = NormalizedSpectrumKernel(p)
        #gram = sk.gram(X)
        #mat = sk.matrix(Y,X)
        #clf = svm.SVC(kernel='precomputed', C=opt_C)
        #clf.fit(gram, label)
        #predict = clf.predict(mat)
        ## </SVM>

        ## <Differenterrorcosts>
        sk = NormalizedSpectrumKernel(p)
        gram = sk.gram(X)
        mat = sk.matrix(Y,X)
        clf = DifferentErrorCosts(sk)
        clf.fit(X, label, C=opt_C, gram=gram)
        predict = clf.predict(mat, precomputed=True)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        #sk = NormalizedSpectrumKernel(p)
        #gram, mat, weight = precompute(sk, X, label, Y)
        #clf = KernelProbabilityFuzzySVM(sk)
        #clf.fit(X, label, C=opt_C, gram=gram, sample_weight=weight)
        #predict = clf.predict(mat, precomputed=True)
        ## </Kernelprobabilityfuzzysvm>

        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "%s[%d-spec] acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (dataname,p,acc,accP,accN,g,_g)

if __name__ == '__main__':
    spam = Dataset("data/SMSSpamCollection.rplcd", isNonvectorial=True, delimiter='\t', dtype={'names':('0','1'), 'formats':('f8','S512')})
    label = spam.raw['0']
    X = spam.raw['1']

    seed = 0
    p = 2
    ratio = [1,2,5,10,20,50,100]
    max_ratio = max(ratio)
    class_label = [1,-1]

    pData = X[label[:]==class_label[0]]
    nData = X[label[:]==class_label[1]]
    pN, nN = len(pData), len(nData)

    if pN > nN:
        major, minor = pData, nData
        maj_N, min_N = pN, nN
        maj_label, min_label = class_label[0], class_label[1]
    else:
        major, minor = nData, pData
        maj_N, min_N = nN, pN
        maj_label, min_label = class_label[1], class_label[0]

    np.random.seed(seed)
    np.random.shuffle(minor)
    np.random.shuffle(major)

    N = np.round(maj_N / max_ratio)
    minor = minor[:N]

    for r in ratio:
        M = N*r
        X = np.r_[major[:M], minor]
        label = np.r_[[maj_label]*M, [min_label]*N]
        procedure("ratio:%d" % r, X, label, p, nCV=5)
