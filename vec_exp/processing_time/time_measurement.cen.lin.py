#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm
from dataset import *
import multiprocessing
import time

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

def dist_from_estimated_hyperplane(X, label, beta):
    t_slice = float(0)

    t_start = time.clock() #----- TIMER START -----

    # sort given sample with their label
    dataset = np.c_[label, X]
    dataset = dataset[dataset[:,0].argsort()]
    label, X = dataset[:,0], dataset[:,1:]

    t_slice += time.clock() - t_start #----- TIMER END -----

    # calc. gram matrix and then sample_weight
    kernel = GaussKernel(beta)
    gram = kernel.gram(X)

    t_start = time.clock() #----- TIMER START -----

    distance= np.dot(np.diag(label), np.dot(gram, label))

    t_slice += time.clock() - t_start #----- TIMER END -----

    return (X, label, gram, distance, t_slice)

def dist_from_hyperplane(X, label, beta, C=1.):
    t_slice = float(0)

    t_start = time.clock() #----- TIMER START -----

    # train conventional SVM
    clf = svm.SVC(kernel='rbf', gamma=beta, C=C)
    clf.fit(X, label)

    # calc. distance between from hyperplane
    value = (clf.decision_function(X))[:,0]
    distance = np.abs(value)

    t_slice += time.clock() - t_start #----- TIMER END -----

    return (distance, t_slice)

def multiproc(args):
    rough_C, gamma_list, beta, Y, answer, X, label = args

    t_slice = float(0)

    #dist_from_center() rearrange the order of samples.
    #so we have to use gram matrix caluclated after rearrangement
    #<FSVMCIL.CENTER>
    kernel = GaussKernel(beta)
    t_start = time.clock() #----- TIMER START -----
    X, label, distance = dist_from_center(X, label)
    t_slice += time.clock() - t_start #----- TIMER END -----
    gram = kernel.gram(X)
    #</FSVMCIL.CENTER>

    #<FSVMCIL.HYPERPLANE>
    #kernel = GaussKernel(beta)
    #gram = kernel.gram(X)
    #</FSVMCIL.HYPERPLANE>

    #dist_from_estimated_hyperplane() rearrange the order of samples.
    #so we have to use gram matrix returned by that method at clf.fit()
    #<FSVMCIL.ESTIMATE>
    ##----- TIMER START -----
    #X, label, gram, distance, t = dist_from_estimated_hyperplane(X, label, beta)
    #t_slice += t
    ##----- TIMER END -----
    #</FSVMCIL.ESTIMATE>

    res = []
    for _C in rough_C:

        #dist_from_hyperplane() doesn't rearange the order of samples,
        #so we can use gram matrix calculated above at clf.fit().
        #<FSVMCIL.HYPERPLANE>
        ##----- TIMER START -----
        #distance, t = dist_from_hyperplane(X, label, beta, _C)
        #t_slice += t
        ##----- TIMER END -----
        #</FSVMCIL.HYPERPLANE>

        #<FSVMCIL.LIN>
        clf = FSVMCIL(beta, distance_function="center", decay_function="linear", delta=0.000001)
        #clf = FSVMCIL(beta, distance_function="estimate", decay_function="linear", delta=0.000001)
        #clf = FSVMCIL(beta, distance_function="hyperplane", decay_function="linear", delta=0.000001)

        t_start = time.clock() #----- TIMER START -----
        weight = clf.linear_decay_function(distance)
        t_slice += time.clock() - t_start #----- TIMER END -----

        predict = np.ones_like(answer)
        res.append( (_C,)+evaluation(predict, answer) )
        #</FSVMCIL.LIN>

    res.append(t_slice)
    return res

def procedure(dataname, dataset, nCV=5, **kwargs):
    t_slice = float(0)

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

            for i,r in enumerate(res):
                res[i], t = r[:-1], r[-1]
                t_slice += t

            #<FSVMCIL.LIN>
            res_foreach_dataset = np.array(res)
            res_foreach_C = np.average(res_foreach_dataset, axis=0)

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta
            #</FSVMCIL.LIN>

        print "[rough search] opt_beta:%s,\topt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_beta,opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        # narrow parameter search
        narrow_C = [opt_C*(10**j) for j in narrow_space]
        for beta in [opt_beta*(10**i) for i in narrow_space]:
            args = [ (narrow_C, gamma_list, beta) + elem for elem in dataset_iterator(pseudo, nCV) ]
            res = pool.map(multiproc, args)

            for i,r in enumerate(res):
                res[i], t = r[:-1], r[-1]
                t_slice += t

            #<FSVMCIL.LIN>
            res_foreach_dataset = np.array(res)
            res_foreach_C = np.average(res_foreach_dataset, axis=0)

            for _C, _acc, _accP, _accN, _g in res_foreach_C:
                _g = np.sqrt(_accP * _accN)
                if _g > max_g: max_g, opt_C, opt_beta = _g, _C, beta
            #</FSVMCIL.LIN>

        print "[narrow search] opt_beta:%s,\topt_C:%s,\topt_gamma:%s,\tg:%f" % (opt_beta,opt_C,opt_gamma,max_g)
        sys.stdout.flush()

        predict = np.ones_like(answer)
        e = evaluation(predict, answer)
        print "[optimized] acc:%f,\taccP:%f,\taccN:%f,\tg:%f" % e
        scores.append(e)

    # average evaluation score
    acc, accP, accN, g = np.average(np.array(scores), axis=0)
    _g = np.sqrt(accP * accN)
    print "[%s]: acc:%f,\taccP:%f,\taccN:%f,\tg:%f,\tg_from_ave.:%f" % (dataname,acc,accP,accN,g,_g)

    return t_slice

if __name__ == '__main__':
#    setting="cen.exp"
    setting="cen.lin"
#    setting="est.exp"
#    setting="est.lin"
#    setting="hyp.exp"
#    setting="hyp.lin"
    with open("processing_time.%s.log" % setting, "w") as fp:
        t_total_start = time.clock()

        ecoli = Dataset("data/ecoli.rplcd", label_index=-1, usecols=range(1,9), dtype=np.float)
        t = procedure('ecoli', ecoli.raw, label_index=-1)
        fp.write('ecoli:%s\n' % t)
        fp.flush()

        transfusion = Dataset("data/transfusion.rplcd", label_index=-1, delimiter=',', skiprows=1, dtype=np.float)
        t = procedure('transfusion', transfusion.raw, label_index=-1)
        fp.write('transfusion:%s\n' % t)
        fp.flush()

        haberman = Dataset("data/haberman.rplcd", label_index=-1, delimiter=',', dtype=np.float)
        t = procedure('haberman', haberman.raw, label_index=-1)
        fp.write('haberman:%s\n' % t)
        fp.flush()

        pima = Dataset("data/pima-indians-diabetes.rplcd", label_index=-1, delimiter=',', dtype=np.float)
        t = procedure('pima', pima.raw, label_index=-1)
        fp.write('pima:%s\n' % t)
        fp.flush()

        yeast = Dataset("data/yeast.rplcd", label_index=-1, usecols=range(1,10), dtype=np.float)
        t = procedure('yeast', yeast.raw, label_index=-1)
        fp.write('yeast:%s\n' % t)
        fp.flush()

        page = Dataset("data/page-blocks.rplcd", label_index=-1, dtype=np.float)
        t = procedure('page-block', page.raw, label_index=-1)
        fp.write('page-block:%s\n' % t)
        fp.flush()

        abalone = Dataset("data/abalone.rplcd", label_index=-1, usecols=range(1,9), delimiter=',', dtype=np.float)
        t = procedure('abalone', abalone.raw, label_index=-1)
        fp.write('abalone:%s\n' % t)
        fp.flush()

        waveform = Dataset("data/waveform.rplcd", label_index=-1, delimiter=',', dtype=np.float)
        t = procedure('waveform', waveform.raw, label_index=-1)
        fp.write('waveform:%s\n' % t)
        fp.flush()

        satimage = Dataset("data/satimage.rplcd", label_index=-1, delimiter=' ', dtype=np.float)
        t = procedure('satimage', satimage.raw, label_index=-1)
        fp.write('satimage:%s\n' % t)
        fp.flush()

        t_total_end = time.clock()
        fp.write('--- total:%s\n ---' % str(t_total_end - t_total_start))
