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

class NoisyDistributionModel:
    def __init__(self, kernel):
        self.kernel = kernel

    def precompute(self, sample, label):
        gram = self.kernel.gram(sample)
        fx = np.dot(np.diag(label), np.dot(gram, label))

        return (gram, fx)

    def px(self, hx, sigma, d, ht_part=0, hc_part=0):
        hx_sorted = np.sort(hx)
        hc = hx_sorted[-int(np.round(len(hx)*hc_part)+1)]
        ht = hx_sorted[int(np.round(len(hx)*ht_part))]

        def f(x):
            if x > hc:
                return 1.
            elif x < ht:
                return sigma
            else:
                ret = (1-sigma)*((x-ht)/(hc-ht))**d
                return sigma + ret

        return (np.vectorize(f))(hx)

    def fit(self, sample, label, C=1., sigma=0.5, d=1, ht_part=0, hc_part=0, gram=None, sample_weight=None):
        self.sample = sample

        # NOT precomputed
        if gram is None or sample_weight is None:
            gram, fx = self.precompute(sample, label)
            sample_weight = self.px(fx, sigma, d, ht_part, hc_part)

        # ready and fit SVM
        self.clf = svm.SVC(kernel='precomputed', C=C)
        self.clf.fit(gram, label, sample_weight=sample_weight)

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample)
        return self.clf.predict(mat)

def search_sigma_d(args):
    sigma, d_list, opt_beta, opt_C, Y, answer, X, label = args

    gk = GaussKernel(opt_beta)
    clf = NoisyDistributionModel(gk)
    gram, fx = clf.precompute(X, label)

    res = []
    for d in d_list:
        weight = clf.px(fx, sigma, d)
        clf.fit(X, label, C=opt_C, gram=gram, sample_weight=weight)

        predict = clf.predict(Y)
        res.append( (d,)+evaluation(predict, answer) )

    return res

def search_part(args):
    ht_part, part_list, opt_beta, opt_C, opt_sigma, opt_d, Y, answer, X, label = args

    gk = GaussKernel(opt_beta)
    clf = NoisyDistributionModel(gk)
    gram, fx = clf.precompute(X, label)

    res = []
    for hc_part in part_list:
        weight = clf.px(fx, opt_sigma, opt_d, ht_part, hc_part)
        clf.fit(X, label, C=opt_C, gram=gram, sample_weight=weight)

        predict = clf.predict(Y)
        res.append( (hc_part,)+evaluation(predict, answer) )

    return res

def multiproc(args):
    rough_C, beta, Y, answer, X, label = args

    res = []
    for _C in rough_C:
        clf = svm.SVC(kernel='rbf', gamma=beta, C=_C)
        clf.fit(X, label)
        predict = clf.predict(Y)
        res.append( (_C,)+evaluation(predict, answer) )

    return res

def search_svm_params(dataset, nCV):
    # define search region
    rough_C = [10**i for i in range(10)]
    rough_beta = [10**i for i in range(-9,1)]
    narrow_space = np.linspace(-0.75, 0.75, num=7)

    # ready parametersearch
    opt_beta, opt_C = 0., 0.

    # rough parameter search
    max_g = -999.
    pool = multiprocessing.Pool(nCV)
    for beta in rough_beta:
        args = [ (rough_C, beta) + elem for elem in dataset_iterator(dataset, nCV) ]
        res = pool.map(multiproc, args)

        res_foreach_dataset = np.array(res)
        res_foreach_C = np.average(res_foreach_dataset, axis=0)

        for _C, _acc, _accP, _accN, _g in res_foreach_C:
            _g = np.sqrt(_accP * _accN)
            if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta

    print "[rough search] opt_beta:%s,\topt_C:%s,\tg:%f" % (opt_beta,opt_C,max_g)
    sys.stdout.flush()
    pool.close()

    # narrow parameter search
    max_g = -999.
    pool = multiprocessing.Pool(nCV)
    narrow_C = [opt_C*(10**j) for j in narrow_space]
    for beta in [opt_beta*(10**i) for i in narrow_space]:
        args = [ (narrow_C, beta) + elem for elem in dataset_iterator(dataset, nCV) ]
        res = pool.map(multiproc, args)

        res_foreach_dataset = np.array(res)
        res_foreach_C = np.average(res_foreach_dataset, axis=0)

        for _C, _acc, _accP, _accN, _g in res_foreach_C:
            _g = np.sqrt(_accP * _accN)
            if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta

    print "[narrow search] opt_beta:%s,\topt_C:%s,\tg:%f" % (opt_beta,opt_C,max_g)
    sys.stdout.flush()
    pool.close()

    return (opt_beta, opt_C)

def search_px_params(dataset, opt_beta, opt_C, nCV):
    # define search region
    sigma_list = np.r_[[0.01], np.arange(0.1,0.9+0.1,0.1)]
    d_list = [2**i for i in range(-8,8+1)]

    # ready parametersearch
    opt_sigma, opt_d = 0., 0.

    max_g = -999.
    pool = multiprocessing.Pool(nCV)
    for sigma in sigma_list:
        args = [ (sigma, d_list, opt_beta, opt_C) + elem for elem in dataset_iterator(dataset, nCV) ]
        res = pool.map(search_sigma_d, args)

        res_foreach_dataset = np.array(res)
        res_foreach_d = np.average(res_foreach_dataset, axis=0)

        for _d, _acc, _accp, _accn, _g in res_foreach_d:
            _g = np.sqrt(_accp * _accn)
            if _g > max_g: max_g, opt_sigma, opt_d = _g, sigma, _d

    print "[sigma,d search] opt_sigma:%s,\topt_d:%s,\tg:%f" % (opt_sigma,opt_d,max_g)
    sys.stdout.flush()
    pool.close()

    return (opt_sigma, opt_d)

def search_hthc_part(dataset, opt_beta, opt_C, opt_sigma, opt_d, nCV):
    # define search region
    part_list = np.arange(0,0.5+0.1,0.1)

    # ready parametersearch
    opt_ht_part, opt_ht_part = 0., 0.

    max_g = -999.
    pool = multiprocessing.Pool(nCV)
    for ht_part in part_list:
        params = (ht_part, part_list, opt_beta, opt_C, opt_sigma, opt_d)
        args = [  params + elem for elem in dataset_iterator(dataset, nCV) ]
        res = pool.map(search_part, args)

        res_foreach_dataset = np.array(res)
        res_foreach_ht_part = np.average(res_foreach_dataset, axis=0)

        for _hc_part, _acc, _accP, _accN, _g in res_foreach_ht_part:
            _g = np.sqrt(_accP * _accN)
            if _g > max_g: max_g, opt_ht_part, opt_hc_part = _g, ht_part, _hc_part

    print "[part search] opt_ht_part:%s,\topt_hc_part:%s,\tg:%f" % (opt_ht_part, opt_hc_part, max_g)
    sys.stdout.flush()
    pool.close()

    return (opt_ht_part, opt_hc_part)

def procedure(dataname, dataset, nCV=5, **kwargs):
    # define search region
    part_list = np.arange(0,0.5+0.1,0.1)

    # cross varidation
    scores = []
    for i_cv, (Y,answer,X,label) in enumerate( dataset_iterator(dataset, nCV, **kwargs) ):
        pos, neg = len(label[label[:]==1]),len(label[label[:]==-1])
        print "%s(%d/%d): train samples (pos:%d, neg:%d)" % (dataname, i_cv, nCV, pos, neg)
        pos, neg = len(answer[answer[:]==1]),len(answer[answer[:]==-1])
        print "%s(%d/%d): test samples (pos:%d, neg:%d)" % (dataname, i_cv, nCV, pos, neg)

        # ready parametersearch
        pseudo = np.c_[label, X]

        # SVM(GaussKernel) parameter search
        opt_beta, opt_C = search_svm_params(pseudo, nCV)

        # Px(x) parameter search
        opt_sigma, opt_d = search_px_params(pseudo, opt_beta, opt_C, nCV)

        # Ht, Hc search
        opt_ht_part, opt_hc_part = search_hthc_part(pseudo, opt_beta, opt_C, opt_sigma, opt_d, nCV)

        # classify using searched params
        gk = GaussKernel(opt_beta)
        clf = NoisyDistributionModel(gk)
        clf.fit(X, label, sigma=opt_sigma, d=opt_d, ht_part=opt_ht_part, hc_part=opt_hc_part)

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
    id = ImbalancedData(posDist, negDist, 10.)
    dataset = id.getSample(500)
    procedure('gaussian mix.', dataset, nCV=4, label_index=0)
