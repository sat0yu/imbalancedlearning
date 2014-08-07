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

def multiproc(args):
    rough_C, beta, Y, answer, X, label = args

    t_slice = float(0)

    ## <Kernelprobabilityfuzzysvm>
    gk = GaussKernel(beta)
    clf = KernelProbabilityFuzzySVM(gk)
    #----- TIMER START -----
    X, gram, label, weight, t = clf.precompute(X, label)
    t_slice += t
    #----- TIMER END -----
    ## </Kernelprobabilityfuzzysvm>


    res = []
    for _C in rough_C:
        ## <Kernelprobabilityfuzzysvm>
        predict = np.ones_like(answer)
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
    setting="proposed"
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

        t_total_end = time.clock()
        fp.write('--- total:%s\n ---' % str(t_total_end - t_total_start))
