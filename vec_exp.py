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

    #clf = FSVMCIL(beta, distance_function="center", decay_function="linear", delta=0.000001)
    clf = FSVMCIL(beta, distance_function="estimate", decay_function="linear", delta=0.000001)
    #clf = FSVMCIL(beta, distance_function="hyperplane", decay_function="linear", delta=0.000001)
    #clf = KernelProbabilityFuzzySVM( GaussKernel(beta) )
    #clf = DifferentErrorCosts( GaussKernel(beta) )
    #X, gram, label, weight = clf.precompute(X, label)
    #gram = clf.precompute(X)

    res = []
    for _C in rough_C:
        clf.fit(X, label, C=_C)
        #clf.fit(X, label, C=_C, gram=gram)
        predict = clf.predict(Y)
        res.append( (_C,)+evaluation(predict, answer) )

    return res

def procedure(dataname, dataset, ratio, nCV=5, **kwargs):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    rough_beta = [10**i for i in range(-9,1)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)

    dataset = createImbalanceClassDataset(dataset, ratio)

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

        print "[narrow search] opt_beta:%s,\topt_C:%s,\tg:%f" % (opt_beta,opt_C,max_g)
        sys.stdout.flush()

        # classify using searched params
        #clf = FSVMCIL(opt_beta, distance_function="center", decay_function="linear", delta=0.000001)
        clf = FSVMCIL(opt_beta, distance_function="estimate", decay_function="linear", delta=0.000001)
        #clf = FSVMCIL(opt_beta, distance_function="hyperplane", decay_function="linear", delta=0.000001)
        #gk = GaussKernel(opt_beta)
        #clf = DifferentErrorCosts(gk)
        #clf = KernelProbabilityFuzzySVM(gk)
        clf.fit(X, label, C=opt_C)
        predict = clf.predict(Y)
        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "[%s]: acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (dataname,acc,accP,accN,g,_g)

def createTwoClassDataset(variances, means_distance, N, ratio, seed=0):
    for v in variances:
        if v.shape[0] is not v.shape[1]:
            raise ValueError("Variance-covariance matrix must have the shape like square matrix")

    r = np.sqrt(means_distance)
    d = variances[0].shape[0]
    np.random.seed(seed)

    posMean = np.r_[r, [0]*(d-1)]
    negMean = np.r_[-r, [0]*(d-1)]

    posDist = NormalDistribution(posMean, variances[0])
    negDist = NormalDistribution(negMean, variances[1])
    imbdata = ImbalancedData(posDist, negDist, ratio)
    dataset = imbdata.getSample(N)

    return dataset

if __name__ == '__main__':
    seed = 0
    ratio = [1,2,5,10,20,50,100]
    raw_ratio = 100
    N = 5000
    dim = 5
    var = np.sqrt(dim)
    dataset = createTwoClassDataset([(var**2)*np.identity(dim)]*2, 2*var, N, raw_ratio, seed=0)

    for r in ratio:
        procedure("dim:%d, var:%.3f, ratio:%d" % (dim,var,r), dataset, r, nCV=5, label_index=0)
