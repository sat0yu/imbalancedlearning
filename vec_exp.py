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
    rough_C, beta, Y, answer, X, label = args

    ## <Differenterrorcosts>
    #gk = GaussKernel(beta)
    #clf = DifferentErrorCosts(gk)
    #gram = gk.gram(X)
    #mat = gk.matrix(Y,X)
    ## </Differenterrorcosts>

    ## <Kernelprobabilityfuzzysvm>
    gk = GaussKernel(beta)
    clf = KernelProbabilityFuzzySVM(gk)
    X, gram, label, weight = clf.precompute(X, label)
    mat = gk.matrix(Y,X)
    ## </Kernelprobabilityfuzzysvm>

    res = []
    for _C in rough_C:
        ## <SVM>
        clf = svm.SVC(kernel='rbf', gamma=beta, C=_C)
        clf.fit(X, label)
        predict = clf.predict(Y)
        ## </SVM>

        ## <Differenterrorcosts>
        #clf.fit(X, label, C=_C, gram=gram)
        #predict = clf.predict(mat, precomputed=True)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        clf.fit(X, label, C=_C, gram=gram, sample_weight=weight)
        predict = clf.predict(mat, precomputed=True)
        ## </Kernelprobabilityfuzzysvm>

        res.append( (_C,)+evaluation(predict, answer) )

    return res

def procedure(dataname, dataset, nCV=5, **kwargs):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    rough_beta = [10**i for i in range(-9,1)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)

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

        print "[rough search] opt_beta:%s,\topt_C:%s,\tg:%f" % (opt_beta,opt_C,max_g)
        sys.stdout.flush()

        # narrow parameter search
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

        print "[narrow search] opt_beta:%s,\topt_C:%s,\tg:%f" % (opt_beta,opt_C,max_g)
        sys.stdout.flush()

        # classify using searched params

        ## <SVM>
        clf = svm.SVC(kernel='rbf', gamma=opt_beta, C=opt_C)
        clf.fit(X, label)
        ## </SVM>

        ## <Differenterrorcosts>
        #clf = DifferentErrorCosts( GaussKernel(opt_beta) )
        #clf.fit(X, label, C=opt_C)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        clf = KernelProbabilityFuzzySVM( GaussKernel(opt_beta) )
        clf.fit(X, label, C=opt_C)
        ## </Kernelprobabilityfuzzysvm>

        predict = clf.predict(Y)
        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "[%s]: acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (dataname,acc,accP,accN,g,_g)

if __name__ == '__main__':
    ecoli = Dataset("data/ecoli.rplcd", label_index=-1, usecols=range(1,9), dtype=np.float)
    ecoli.raw = np.c_[ecoli.normalize(), ecoli.label]
    procedure('ecoli', ecoli.raw, label_index=-1)

    transfusion = Dataset("data/transfusion.rplcd", label_index=-1, delimiter=',', skiprows=1, dtype=np.float)
    transfusion.raw = np.c_[transfusion.normalize(), transfusion.label]
    procedure('transfusion', transfusion.raw, label_index=-1)

    haberman = Dataset("data/haberman.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    haberman.raw = np.c_[haberman.normalize(), haberman.label]
    procedure('haberman', haberman.raw, label_index=-1)

    pima = Dataset("data/pima-indians-diabetes.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    pima.raw = np.c_[pima.normalize(), pima.label]
    procedure('pima', pima.raw, label_index=-1)

    yeast = Dataset("data/yeast.rplcd", label_index=-1, usecols=range(1,10), dtype=np.float)
    yeast.raw = np.c_[yeast.normalize(), yeast.label]
    procedure('yeast', yeast.raw, label_index=-1)

    page = Dataset("data/page-blocks.rplcd", label_index=-1, dtype=np.float)
    page.raw = np.c_[page.normalize(), page.label]
    procedure('page-block', page.raw, label_index=-1)

    abalone = Dataset("data/abalone.rplcd", label_index=-1, usecols=range(1,9), delimiter=',', dtype=np.float)
    abalone.raw = np.c_[abalone.normalize(), abalone.label]
    procedure('abalone', abalone.raw, label_index=-1)

    waveform = Dataset("data/waveform.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    waveform.raw = np.c_[waveform.normalize(), waveform.label]
    procedure('waveform', waveform.raw, label_index=-1)
