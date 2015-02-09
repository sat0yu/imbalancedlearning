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

def multiproc(args):
    rough_C, ngram_param, Y, answer, X, label = args

    ## <SVM>
    # opk = NormalizedOPNgramKernel(ngram_param)
    # gram = opk.gram(X)
    # mat = opk.matrix(Y,X)
    ## </SVM>

    ## <Differenterrorcosts>
    opk = NormalizedOPNgramKernel(ngram_param)
    clf = DifferentErrorCosts(opk)
    gram = clf.precompute(X)
    mat = opk.matrix(Y,X)
    ## </Differenterrorcosts>

    ## <Kernelprobabilityfuzzysvm>
    # opk = NormalizedOPNgramKernel(ngram_param)
    # clf = KernelProbabilityFuzzySVM(opk)
    # X, gram, label, weight = clf.precompute(X, label)
    # mat = opk.matrix(Y,X)
    ## </Kernelprobabilityfuzzysvm>

    res = []
    for _C in rough_C:
        ## <SVM>
        # clf = svm.SVC(kernel='precomputed', C=_C)
        # clf.fit(gram, label)
        # predict = clf.predict(mat)
        ## </SVM>

        ## <Differenterrorcosts>
        clf.fit(X, label, C=_C, gram=gram)
        predict = clf.predict(mat, precomputed=True)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        # clf.fit(X, label, C=_C, gram=gram, sample_weight=weight)
        # predict = clf.predict(mat, precomputed=True)
        ## </Kernelprobabilityfuzzysvm>

        print("LOG: ",  (_C,)+evaluation(predict, answer) )
        res.append( (_C,)+evaluation(predict, answer) )

    return res

def procedure(dataname, dataset, nCV=5, **kwargs):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)
    ngram_param = 4
    # cross varidation
    scores = []
    for i_CV, (Y,answer,X,label) in enumerate( dataset_iterator(dataset, nCV, **kwargs) ):
        pos, neg = len(label[label[:]==1]),len(label[label[:]==-1])
        print "%s(%d/%d): train samples (pos:%d, neg:%d)" % (dataname, i_CV, nCV, pos, neg)
        pos, neg = len(answer[answer[:]==1]),len(answer[answer[:]==-1])
        print "%s(%d/%d): test samples (pos:%d, neg:%d)" % (dataname, i_CV, nCV, pos, neg)

        # ready parametersearch
        pseudo = np.c_[label, X]
        pool = multiprocessing.Pool(nCV)
        opt_C, max_g = 0., -999.

        # rough parameter search
        args = [ (rough_C, ngram_param) + elem for elem in dataset_iterator(pseudo, nCV) ]
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
        narrow_C = [opt_C*(10**j) for j in narrow_space]
        args = [ (narrow_C, ngram_param) + elem for elem in dataset_iterator(pseudo, nCV) ]
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
        # opk = NormalizedOPNgramKernel(ngram_param)
        # gram = opk.gram(X)
        # mat = opk.matrix(Y,X)
        # clf = svm.SVC(kernel='precomputed', C=opt_C)
        # clf.fit(gram, label)
        # predict = clf.predict(mat)
        ## </SVM>

        ## <Differenterrorcosts>
        opk = NormalizedOPNgramKernel(ngram_param)
        clf = DifferentErrorCosts(opk)
        clf.fit(X, label, C=opt_C)
        predict = clf.predict(Y)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        # opk = NormalizedOPNgramKernel(ngram_param)
        # clf = KernelProbabilityFuzzySVM(opk)
        # clf.fit(X, label, C=opt_C)
        # predict = clf.predict(Y)
        ## </Kernelprobabilityfuzzysvm>

        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "[%s]: acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (dataname,acc,accP,accN,g,_g)

if __name__ == '__main__':
    lp5 = Dataset("data/lp5.data", label_index=0, delimiter=',', dtype=np.int)
    lp5.raw = np.c_[lp5.data, lp5.label]
    procedure('lp5', lp5.raw, label_index=-1)
