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

def multiproc(args):
    rough_C, gamma_list, p, Y, answer, X, label = args

    #<FSVMCIL_NONVEC.ESTIMATE.LIN>
    #sk = NormalizedSpectrumKernel(p)
    #clf = FSVMCIL_NONVEC(sk, decay_function="linear", delta=0.000001)
    #gram, distance = clf.dist_from_estimated_hyperplane(X, label)
    #mat = clf.matrix(Y, X)
    #weight = clf.decay_function(distance)
    #</FSVMCIL_NONVEC.ESTIMATE.LIN>

    #<FSVMCIL_NONVEC.ESTIMATE.EXP>
    #sk = NormalizedSpectrumKernel(p)
    #temp = FSVMCIL_NONVEC(sk, decay_function="exp")
    #gram, distance = temp.dist_from_estimated_hyperplane(X, label)
    #mat = temp.matrix(Y, X)
    #</FSVMCIL_NONVEC.ESTIMATE.EXP>

    #<FSVMCIL_NONVEC.HYPERPLANE.LIN>
    sk = NormalizedSpectrumKernel(p)
    clf = FSVMCIL_NONVEC(sk, decay_function="linear", delta=0.000001)
    gram = clf.gram(X)
    mat = clf.matrix(Y, X)
    #</FSVMCIL_NONVEC.HYPERPLANE.LIN>

    res = []
    for _C in rough_C:
        #<FSVMCIL_NONVEC.ESTIMATE.LIN>
        #clf.fit(gram, label, weight, C=_C)
        #predict = clf.predict(mat)
        #res.append( (_C,)+evaluation(predict, answer) )
        #</FSVMCIL_NONVEC.ESTIMATE.LIN>

        #<FSVMCIL_NONVEC.HYPERPLANE.LIN>
        distance = clf.dist_from_hyperplane(gram, label)
        weight = clf.decay_function(distance)
        clf.fit(gram, label, weight, C=_C)
        predict = clf.predict(mat)
        res.append( (_C,)+evaluation(predict, answer) )
        #</FSVMCIL_NONVEC.HYPERPLANE.LIN>

        #<FSVMCIL_NONVEC.EXP>
        #for _g in gamma_list:
        #    clf = FSVMCIL_NONVEC(sk, decay_function="exp", gamma=_g)
        #    weight = clf.decay_function(distance)
        #    clf.fit(gram, label, weight, C=_C)
        #    predict = clf.predict(mat)
        #    res.append( (_C,_g)+evaluation(predict, answer) )
        #</FSVMCIL_NONVEC.EXP>

    return res

def procedure(stringdata, datalabel, p, nCV=5):
    # ready parameter search space
    rough_C = [10**i for i in range(10)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)
    gamma_list = np.linspace(0.1, 1.0, 10)

    # cross varidation
    scores = []
    for i_CV, (Y,answer,X,label) in enumerate( dataset_iterator(stringdata, datalabel, nCV) ):
        pos, neg = len(label[label[:]==1]),len(label[label[:]==-1])
        print "[%d/%d]: train samples (pos:%d, neg:%d)" % (i_CV, nCV, pos, neg)
        pos, neg = len(answer[answer[:]==1]),len(answer[answer[:]==-1])
        print "[%d/%d]: test samples (pos:%d, neg:%d)" % (i_CV, nCV, pos, neg)

        # ready parametersearch
        pool = multiprocessing.Pool(5)
        opt_C, opt_gamma, max_g = 0., 0., -999.

        # rough parameter search
        args = [ (rough_C, gamma_list, p) + elem for elem in dataset_iterator(X, label, nCV) ]
        res = pool.map(multiproc, args)

        #<FSVMCIL_NONVEC.LIN>
        res_foreach_dataset = np.array(res)
        res_foreach_C = np.average(res_foreach_dataset, axis=0)

        for _C, _acc, _accP, _accN, _g in res_foreach_C:
            _g = np.sqrt(_accP * _accN)
            if _g > max_g: max_g, opt_C  = _g, _C
        #</FSVMCIL_NONVEC.LIN>

        #<FSVMCIL_NONVEC.EXP>
        #res_foreach_dataset = np.array(res)
        #res_foreach_C_gamma = np.average(res_foreach_dataset, axis=0)

        #for _C, _gamma, _acc, _accP, _accN, _g in res_foreach_C_gamma:
        #    _g = np.sqrt(_accP * _accN)
        #    if _g > max_g: max_g, opt_C, opt_gamma  = _g, _C, _gamma
        #</FSVMCIL_NONVEC.EXP>

        print "[rough search] opt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        # narrow parameter search
        narrow_C = [opt_C*(10**j) for j in narrow_space]
        args = [ (narrow_C, gamma_list, p) + elem for elem in dataset_iterator(X, label, nCV) ]
        res = pool.map(multiproc, args)

        #<FSVMCIL_NONVEC.LIN>
        res_foreach_dataset = np.array(res)
        res_foreach_C = np.average(res_foreach_dataset, axis=0)

        for _C, _acc, _accP, _accN, _g in res_foreach_C:
            _g = np.sqrt(_accP * _accN)
            if _g > max_g: max_g, opt_C = _g, _C
        #</FSVMCIL_NONVEC.LIN>

        #<FSVMCIL_NONVEC.EXP>
        #res_foreach_dataset = np.array(res)
        #res_foreach_C_gamma = np.average(res_foreach_dataset, axis=0)

        #for _C, _gamma, _acc, _accP, _accN, _g in res_foreach_C_gamma:
        #    _g = np.sqrt(_accP * _accN)
        #    if _g > max_g: max_g, opt_C, opt_gamma = _g, _C, _gamma
        #</FSVMCIL_NONVEC.EXP>

        print "[narrow search] opt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        # classify using searched params
        #<FSVMCIL_NONVEC.LIN>
        #sk = NormalizedSpectrumKernel(p)
        #clf = FSVMCIL_NONVEC(sk, decay_function="linear", delta=0.000001)
        #gram, distance = clf.dist_from_estimated_hyperplane(X, label)
        #mat = clf.matrix(Y, X)
        #weight = clf.decay_function(distance)
        #</FSVMCIL_NONVEC.LIN>

        #<FSVMCIL_NONVEC.EXP>
        #sk = NormalizedSpectrumKernel(p)
        #clf = FSVMCIL_NONVEC(sk, decay_function="exp", gamma=opt_gamma)
        #gram, distance = clf.dist_from_estimated_hyperplane(X, label)
        #mat = clf.matrix(Y, X)
        #weight = clf.decay_function(distance)
        #</FSVMCIL_NONVEC.EXP>

        #<FSVMCIL_NONVEC.LIN>
        sk = NormalizedSpectrumKernel(p)
        clf = FSVMCIL_NONVEC(sk, decay_function="linear", delta=0.000001)
        gram = clf.gram(X)
        mat = clf.matrix(Y, X)
        distance = clf.dist_from_hyperplane(gram, label)
        weight = clf.decay_function(distance)
        #</FSVMCIL_NONVEC.LIN>

        clf.fit(gram, label, weight, C=opt_C)
        predict = clf.predict(mat)
        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "[%d-spec] acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (p,acc,accP,accN,g,_g)

if __name__ == '__main__':
    spam = Dataset("data/SMSSpamCollection.rplcd.reduced", isNonvectorial=True, delimiter='\t', dtype={'names':('0','1'), 'formats':('f8','S1024')})
    label = spam.raw['0']
    X = spam.raw['1']
    procedure(X, label, 2, nCV=5)
