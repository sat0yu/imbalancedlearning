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

def dist_from_center(X, label, class_label=[1,-1]):
    def dist(_X):
        # calc. center
        center = np.average(_X, axis=0)
        # calc. distance between from center for each sample
        return np.sum(np.abs(_X - center)**2, axis=-1)**(1/2.)

    # sort given sample with their label
    dataset = np.c_[label, X]
    dataset = dataset[dataset[:,0].argsort()]
    label, X = dataset[:,0], dataset[:,1:]

    # separate given samples according to theirs label
    numNeg = len(label[label[:]==class_label[1]])
    negData, posData  = X[:int(numNeg)], X[int(numNeg):]

    # concatenate arrays
    distance = np.r_[dist(negData), dist(posData)]

    return (X, label, distance)

def dist_from_estimated_hyperplane(X, label, beta, Y):
    # sort given sample with their label
    dataset = np.c_[label, X]
    dataset = dataset[dataset[:,0].argsort()]
    label, X = dataset[:,0], dataset[:,1:]

    # calc. gram matrix and then sample_weight
    kernel = GaussKernel(beta)
    gram = kernel.gram(X)
    distance= np.dot(np.diag(label), np.dot(gram, label))

    mat = kernel.matrix(Y,X)

    return (X, label, gram, mat, distance)

def dist_from_hyperplane(X, label, beta, C=1.):
    # train conventional SVM
    clf = svm.SVC(kernel='rbf', gamma=beta, C=C)
    clf.fit(X, label)

    # calc. distance between from hyperplane
    value = (clf.decision_function(X))[:,0]
    distance = np.abs(value)

    return distance

def multiproc(args):
    rough_C, gamma_list, beta, Y, answer, X, label = args

    #dist_from_center() rearrange the order of samples.
    #so we have to use gram matrix caluclated after rearrangement
    #<FSVMCIL.CENTER>
    #X, label, distance = dist_from_center(X, label)
    #kernel = GaussKernel(beta)
    #gram = kernel.gram(X)
    #mat = kernel.matrix(Y,X)
    #</FSVMCIL.CENTER>

    #<FSVMCIL.HYPERPLANE>
    kernel = GaussKernel(beta)
    gram = kernel.gram(X)
    mat = kernel.matrix(Y,X)
    #</FSVMCIL.HYPERPLANE>

    #dist_from_estimated_hyperplane() rearrange the order of samples.
    #so we have to use gram matrix returned by that method at clf.fit()
    #<FSVMCIL.ESTIMATE>
    #X, label, gram, mat, distance = dist_from_estimated_hyperplane(X, label, beta, Y)
    #</FSVMCIL.ESTIMATE>

    res = []
    for _C in rough_C:

        #dist_from_hyperplane() doesn't rearange the order of samples,
        #so we can use gram matrix calculated above at clf.fit().
        #<FSVMCIL.HYPERPLANE>
        distance = dist_from_hyperplane(X, label, beta, _C)
        #</FSVMCIL.HYPERPLANE>

        #<FSVMCIL.EXP>
        #for _g in gamma_list:
            #clf = FSVMCIL(beta, distance_function="center", decay_function="exp", gamma=_g)
            #clf = FSVMCIL(beta, distance_function="estimate", decay_function="exp", gamma=_g)
            #clf = FSVMCIL(beta, distance_function="hyperplane", decay_function="exp", gamma=_g)

            #weight = clf.exp_decay_function(distance)
            #clf.fit(X, label, C=_C, gram=gram, weight=weight)

            #predict = clf.predict(mat)
            #res.append( (_C,_g)+evaluation(predict, answer) )
        #</FSVMCIL.EXP>

        #<FSVMCIL.LIN>
        #clf = FSVMCIL(beta, distance_function="center", decay_function="linear", delta=0.000001)
        #clf = FSVMCIL(beta, distance_function="estimate", decay_function="linear", delta=0.000001)
        clf = FSVMCIL(beta, distance_function="hyperplane", decay_function="linear", delta=0.000001)

        weight = clf.linear_decay_function(distance)
        clf.fit(X, label, C=_C, gram=gram, weight=weight)

        predict = clf.predict(mat)
        res.append( (_C,)+evaluation(predict, answer) )
        #</FSVMCIL.LIN>

    return res

def procedure(dataname, dataset, ratio, nCV=5, **kwargs):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    rough_beta = [10**i for i in range(-9,1)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)
    gamma_list = np.linspace(0.1, 1.0, 10)

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
        opt_beta, opt_C, opt_gamma, max_g = 0., 0., 0., -999.

        # rough parameter search
        for beta in rough_beta:
            args = [ (rough_C, gamma_list, beta) + elem for elem in dataset_iterator(pseudo, nCV) ]
            res = pool.map(multiproc, args)

            #<FSVMCIL.LIN>
            res_foreach_dataset = np.array(res)
            res_foreach_C = np.average(res_foreach_dataset, axis=0)

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta
            #</FSVMCIL.LIN>

            #<FSVMCIL.EXP>
            #res_foreach_dataset = np.array(res)
            #res_foreach_C_gamma = np.average(res_foreach_dataset, axis=0)

            #for _C, _gamma, _acc, _accP, _accN, _g in res_foreach_C_gamma:
            #    _g = np.sqrt(_accP * _accN)
            #    if _g > max_g: max_g, opt_C, opt_gamma, opt_beta  = _g, _C, _gamma, beta
            #</FSVMCIL.EXP>

        print "[rough search] opt_beta:%s,\topt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_beta,opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        # narrow parameter search
        max_g = -999.
        narrow_C = [opt_C*(10**j) for j in narrow_space]
        for beta in [opt_beta*(10**i) for i in narrow_space]:
            args = [ (narrow_C, gamma_list, beta) + elem for elem in dataset_iterator(pseudo, nCV) ]
            res = pool.map(multiproc, args)

            #<FSVMCIL.LIN>
            res_foreach_dataset = np.array(res)
            res_foreach_C = np.average(res_foreach_dataset, axis=0)

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta
            #</FSVMCIL.LIN>

            #<FSVMCIL.EXP>
            #res_foreach_dataset = np.array(res)
            #res_foreach_C_gamma = np.average(res_foreach_dataset, axis=0)

            #for _C, _gamma, _acc, _accP, _accN, _g in res_foreach_C_gamma:
            #    _g = np.sqrt(_accP * _accN)
            #    if _g > max_g: max_g, opt_C, opt_gamma, opt_beta  = _g, _C, _gamma, beta
            #</FSVMCIL.EXP>

        print "[narrow search] opt_beta:%s,\topt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_beta,opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        # classify using searched params
        #<FSVMCIL.LIN>
        #clf = FSVMCIL(beta, distance_function="center", decay_function="linear", delta=0.000001)
        #clf = FSVMCIL(beta, distance_function="estimate", decay_function="linear", delta=0.000001)
        clf = FSVMCIL(beta, distance_function="hyperplane", decay_function="linear", delta=0.000001)
        #</FSVMCIL.LIN>

        #<FSVMCIL.EXP>
        #clf = FSVMCIL(opt_beta, distance_function="center", decay_function="exp", gamma=opt_gamma)
        #clf = FSVMCIL(opt_beta, distance_function="estimate", decay_function="exp", gamma=opt_gamma)
        #clf = FSVMCIL(opt_beta, distance_function="hyperplane", decay_function="exp", gamma=opt_gamma)
        #</FSVMCIL.EXP>

        clf.fit(X, label, C=opt_C)
        predict = clf.predict(Y)
        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "[%s]: acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (dataname,acc,accP,accN,g,_g)

if __name__ == '__main__':
    seed = 0
    ratio = [1,2,5,10,20,50,100]
    raw_ratio = max(ratio)
    N = 5000
    dim = 5
    var = np.sqrt(dim)
    dataset = createTwoClassDataset([(var**2)*np.identity(dim)]*2, 2*var, N, raw_ratio, seed=0)

    for r in ratio:
        procedure("dim:%d, var:%.3f, ratio:%d" % (dim,var,r), dataset, r, nCV=5, label_index=0)
