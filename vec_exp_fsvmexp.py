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
    rough_C, gamma_list, beta, Y, answer, X, label = args

    res = []
    for _C in rough_C:
        for _g in gamma_list:
            clf = FSVMCIL(beta, distance_function="center", decay_function="exp", gamma=_g)
            #clf = FSVMCIL(beta, distance_function="estimate", decay_function="exp", gamma=_g)
            #clf = FSVMCIL(beta, distance_function="hyperplane", decay_function="exp", gamma=_g)

            clf.fit(X, label, C=_C)
            predict = clf.predict(Y)
            res.append( (_C,_g)+evaluation(predict, answer) )

    return res

def procedure(dataname, dataset, nCV=5, **kwargs):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    rough_beta = [10**i for i in range(-9,1)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)
    gamma_list = np.linspace(0.1, 1.0, 10)

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

            res_foreach_dataset = np.array(res)
            #print res_foreach_dataset.shape
            res_foreach_C_gamma = np.average(res_foreach_dataset, axis=0)
            #print res_foreach_C_gamma.shape

            for _C, _gamma, _acc, _accP, _accN, _g in res_foreach_C_gamma:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_gamma, opt_beta  = _g, _C, _gamma, beta

        print "[rough search] opt_beta:%s,\topt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_beta,opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        # narrow parameter search
        max_g = -999.
        narrow_C = [opt_C*(10**j) for j in narrow_space]
        for beta in [opt_beta*(10**i) for i in narrow_space]:
            args = [ (narrow_C, gamma_list, beta) + elem for elem in dataset_iterator(pseudo, nCV) ]
            res = pool.map(multiproc, args)

            res_foreach_dataset = np.array(res)
            #print res_foreach_dataset.shape
            res_foreach_C_gamma = np.average(res_foreach_dataset, axis=0)
            #print res_foreach_C_gamma.shape

            for _C, _gamma, _acc, _accP, _accN, _g in res_foreach_C_gamma:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_gamma, opt_beta  = _g, _C, _gamma, beta

        print "[narrow search] opt_beta:%s,\topt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_beta,opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        # classify using searched params
        clf = FSVMCIL(opt_beta, distance_function="center", decay_function="exp", gamma=opt_gamma)
        #clf = FSVMCIL(opt_beta, distance_function="estimate", decay_function="exp", gamma=opt_gamma)
        #clf = FSVMCIL(opt_beta, distance_function="hyperplane", decay_function="exp", gamma=opt_gamma)
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
    posDist = NormalDistribution([-10, -10], [[50,0],[0,100]])
    negDist = NormalDistribution([10, 10], [[100,0],[0,50]])
    id = ImbalancedData(posDist, negDist, 5.)
    dataset = id.getSample(500)
    procedure('gaussian mix.', dataset, nCV=4, label_index=0)

    #ecoli = Dataset("data/ecoli.rplcd", label_index=-1, usecols=range(1,9), dtype=np.float)
    #procedure('ecoli', ecoli.raw, label_index=-1)

    #transfusion = Dataset("data/transfusion.rplcd", label_index=-1, delimiter=',', skiprows=1, dtype=np.float)
    #procedure('transfusion', transfusion.raw, label_index=-1)

    #haberman = Dataset("data/haberman.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    #procedure('haberman', haberman.raw, label_index=-1)

    #pima = Dataset("data/pima-indians-diabetes.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    #procedure('pima', pima.raw, label_index=-1)

    #yeast = Dataset("data/yeast.rplcd", label_index=-1, usecols=range(1,10), dtype=np.float)
    #procedure('yeast', yeast.raw, label_index=-1)

    #page = Dataset("data/page-blocks.rplcd", label_index=-1, dtype=np.float)
    #procedure('page-block', page.raw, label_index=-1)

    #abalone = Dataset("data/abalone.rplcd", label_index=-1, usecols=range(1,9), delimiter=',', dtype=np.float)
    #procedure('abalone', abalone.raw, label_index=-1)

    #waveform = Dataset("data/waveform.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    #procedure('waveform', waveform.raw, label_index=-1)
