#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm
from dataset import *
import multiprocessing
import matplotlib.pyplot as plt
import time

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *
from cil import *

def multiproc(args):
    rough_C, beta, Y, answer, X, label = args

    t_slice = float(0)

    ## <SVM>
    gk = GaussKernel(beta)
    gram = gk.gram(X)
    mat = gk.matrix(Y,X)
    ## </SVM>

    ## <Differenterrorcosts>
    #gk = GaussKernel(beta)
    #clf = DifferentErrorCosts(gk)
    #t_start = time.clock() #----- TIMER START -----
    #clf.class_weight(label) # actually, this line is useless
    #t_slice += time.clock() - t_start #----- TIMER END -----
    #gram = gk.gram(X)
    #mat = gk.matrix(Y,X)
    ## </Differenterrorcosts>

    ## <Kernelprobabilityfuzzysvm>
    #gk = GaussKernel(beta)
    #clf = KernelProbabilityFuzzySVM(gk)
    ##----- TIMER START -----
    #X, gram, label, weight, t = clf.precompute(X, label)
    #t_slice += t
    ##----- TIMER END -----
    #mat = gk.matrix(Y,X)
    ## </Kernelprobabilityfuzzysvm>


    res = []
    for _C in rough_C:
        ## <SVM>
        clf = svm.SVC(kernel='precomputed', C=_C)
        clf.fit(gram, label)
        predict = clf.predict(mat)
        ## </SVM>

        ## <Differenterrorcosts>
        #clf.fit(X, label, C=_C, gram=gram)
        #predict = clf.predict(mat, precomputed=True)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        #clf.fit(X, label, C=_C, gram=gram, sample_weight=weight)
        #predict = clf.predict(mat, precomputed=True)
        ## </Kernelprobabilityfuzzysvm>

        res.append( (_C,)+evaluation(predict, answer) )

    res.append(t_slice)
    return res

def procedure(dataname, dataset, nCV=5, **kwargs):
    t_slice = float(0)

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

            for i,r in enumerate(res):
                res[i], t = r[:-1], r[-1]
                t_slice += t

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

            for i,r in enumerate(res):
                res[i], t = r[:-1], r[-1]
                t_slice += t

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
        #clf = svm.SVC(kernel='rbf', gamma=opt_beta, C=opt_C)
        #clf.fit(X, label)
        ## </SVM>

        ## <Differenterrorcosts>
        #clf = DifferentErrorCosts( GaussKernel(opt_beta) )
        #clf.fit(X, label, C=opt_C)
        ## </Differenterrorcosts>

        ## <Kernelprobabilityfuzzysvm>
        #clf = KernelProbabilityFuzzySVM( GaussKernel(opt_beta) )
        #clf.fit(X, label, C=opt_C)
        ## </Kernelprobabilityfuzzysvm>

        predict = clf.predict(Y)
        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "[%s]: acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (dataname,acc,accP,accN,g,_g)

    return t_slice

if __name__ == '__main__':
    np.random.seed(0)
    ratio = [1,2,5,10,20,50,100]
    max_ratio = max(ratio)

    # ----< create artificial dataset >-----
    # the shape of this dataset is like below:
    #         ++
    #   +  x  ++  x  +
    #   +  x  ++  x  +
    #         ++
    N = 5000
    nNeg = int(max_ratio * (N / (1. + max_ratio)))
    nPos = N - nNeg
    nPosInLeft = int(nPos / 2.)
    nPosInRight = nPos - nPosInLeft
    nNegCen = int(nNeg / 2.)
    nNegOutLeft = int((nNeg - nNegCen) / 2.)
    nNegOutRight = nNeg - nNegCen - nNegOutLeft
    dim = 2
    side_var = np.array([[ 0.25, 0.0 ],
                        [ 0.0, 0.5 ]])
    center_var = np.array([[ 0.5, 0.0 ],
                            [ 0.0, 1.0 ]])
    innerWidth, outerWidth = 1.5, 3
    center     = (NormalDistribution(np.zeros(dim), center_var)).create(nNegCen)
    innerLeft  = (NormalDistribution([-1 * innerWidth, 0], side_var)).create(nPosInLeft)
    innerRight = (NormalDistribution([innerWidth, 0], side_var)).create(nPosInRight)
    outerLeft  = (NormalDistribution([-1 * outerWidth, 0], side_var)).create(nNegOutLeft)
    outerRight = (NormalDistribution([outerWidth, 0], side_var)).create(nNegOutRight)
    data = np.r_[center, outerLeft, outerRight, innerLeft, innerRight]
    label = np.array([-1,]*nNeg + [1,]*nPos)
    raw_dataset = np.c_[data, label]
    np.random.shuffle(raw_dataset)
#    print raw_dataset[-100:]
#    plt.scatter(raw_dataset[raw_dataset[:,2]==-1,0], raw_dataset[raw_dataset[:,2]==-1,1], c='b')
#    plt.scatter(raw_dataset[raw_dataset[:,2]==1,0], raw_dataset[raw_dataset[:,2]==1,1], c='r')
#    plt.show()
    # ----</ create artificial dataset >-----

    setting = "svm"
    for r in ratio:
        dataset = createImbalanceClassDataset(raw_dataset, r, label_index=-1)
        t = procedure("ratio:%d" % r, dataset, label_index=-1)
        print "method:%s, ratio:%s, processing time:%s" % (setting, r, t)
