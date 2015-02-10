#coding: utf-8;
import numpy as np
cimport numpy as np

from kernel import *

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

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

def createImbalanceClassDataset(dataset, ratio=1., label_index=0, label=[1,-1]):
    pDataset = dataset[dataset[:,label_index]==label[0]]
    nDataset = dataset[dataset[:,label_index]==label[1]]
    pN, nN = pDataset.shape[0], nDataset.shape[0]

    if pN > nN:
        _pN = np.round(nN * ratio)
        if _pN > pN:
            print "nN * ratio = %d, but positive data has only %d samples" % (_pN, pN)
        pN = _pN
    else:
        _nN = np.round(pN * ratio)
        if _nN > nN:
            print "pN * ratio = %d, but negative data has only %d samples" % (_nN, nN)
        nN = _nN

    return np.r_[nDataset[:nN], pDataset[:pN]]

def dataset_iterator(dataset, nCV, label_index=0, label=[1,-1], shuffle=False):
    # if given shuffle flag
    if shuffle: np.random.shuffle(dataset)

    pDataset = dataset[dataset[:,label_index]==label[0]]
    pw = len(pDataset) / nCV
    nDataset = dataset[dataset[:,label_index]==label[1]]
    nw = len(nDataset) / nCV

    for i in range(nCV):
        pPiv, nPiv = i*pw, i*nw

        # slice out X(Y) from pos/neg dataset
        if i < nCV -1:
            pX = pDataset[pPiv:pPiv+pw]
            nX = nDataset[nPiv:nPiv+nw]
            pY = np.r_[pDataset[:pPiv],pDataset[pPiv+pw:]]
            nY = np.r_[nDataset[:nPiv],nDataset[nPiv+nw:]]
            X, Y = np.r_[pX,nX], np.r_[pY, nY]
        else:
            X = np.r_[pDataset[pPiv:], nDataset[nPiv:]]
            Y = np.r_[pDataset[:pPiv], nDataset[:nPiv]]

        # slice out label(answer) from X(Y)
        lbl, ans = X[:,label_index], Y[:,label_index]

        # slice out train(test)data from X(Y)
        if label_index >= 0:
            traindata = np.c_[X[:,:label_index:], X[:,label_index+1:]]
            testdata = np.c_[Y[:,:label_index:], Y[:,label_index+1:]]
        else:
            # if given label index is negative,
            # forcibly use -1 as index number
            traindata = X[:,:-1]
            testdata = Y[:,:-1]

        yield (traindata,lbl,testdata,ans)

cdef class KernelDensityEstimater:
    cdef np.ndarray sample
    cdef double variance, nConst
    cdef object kernel

    def __init__(self, beta):
        self.variance = 1. / (2 * beta)
        self.kernel = GaussKernel(beta)

    def fit(self, X):
        self.sample = X
        h, dim = X.shape
        n = np.sqrt( 2. * np.pi * self.variance )**dim
        self.nConst = n * h

    cdef double superposition(self, np.ndarray[DTYPE_float_t, ndim=1] x):
        cdef np.ndarray[DTYPE_float_t, ndim=2] sample = self.sample
        cdef np.ndarray[DTYPE_float_t, ndim=1] xi
        return sum([ self.kernel.val(x, xi) for xi in sample ])

    def prob(self, np.ndarray[DTYPE_float_t, ndim=1] x):
        cdef np.ndarray[DTYPE_float_t, ndim=2] sample = self.sample
        cdef np.ndarray[DTYPE_float_t, ndim=1] xi
        return sum([ self.kernel.val(x, xi) for xi in sample ]) / self.nConst

    def estimate(self, np.ndarray[DTYPE_float_t, ndim=2] X):
        cdef np.ndarray[DTYPE_float_t, ndim=1] superposition, xi
        superposition = np.array([ self.superposition(xi) for xi in X ])
        return superposition / self.nConst

def evaluation(predict, answer, posLabel=1, negLabel=-1):
    idxPos = answer[:]==posLabel
    idxNeg = answer[:]==negLabel
    numPos = len(answer[idxPos])
    numNeg = len(answer[idxNeg])

    acc = sum(predict == answer) / float(len(answer))
    accPos = sum(predict[idxPos] == answer[idxPos]) / float(numPos)
    accNeg = sum(predict[idxNeg] == answer[idxNeg]) / float(numNeg)
    g = np.sqrt(accPos * accNeg)

    return (acc,accPos,accNeg,g)

class NormalDistribution():
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def create(self, num):
        return np.random.multivariate_normal(self.mean, self.cov, num)

class ImbalancedData():
    def __init__(self, distPos, distNeg, ratio):
        self.distPos = distPos
        self.distNeg = distNeg
        self.ratio = ratio

    def getSample(self, num, labels=[1,-1], shuffle=False):
        numPos = int( num * ( 1. / (self.ratio + 1.) ) )
        numNeg = num - numPos

        posSample = self.distPos.create(numPos)
        posDataset = np.c_[ [labels[0]]*posSample.shape[0], posSample ]
        negSample = self.distNeg.create(numNeg)
        negDataset = np.c_[ [labels[1]]*negSample.shape[0], negSample ]

        returned = np.r_[posDataset, negDataset]
        if shuffle is True:
            np.random.shuffle(returned)

        return returned

def draw_contour(f, coodinates, *args, plot=None, density=1., **kwargs):
    cdef int p, q
    cdef int x0 = coodinates[0], y0 = coodinates[1]
    cdef int x1 = coodinates[2], y1 = coodinates[3]
    cdef int w = int(density*abs(x1 - x0))
    cdef int h = int(density*abs(y1 - y0))
    cdef np.ndarray[DTYPE_float_t, ndim=2] i,j

    cdef np.ndarray[DTYPE_float_t, ndim=1] I = np.linspace(x0, x1, num = w)
    cdef np.ndarray[DTYPE_float_t, ndim=1] J = np.linspace(y1, y0, num = h)
    cdef np.ndarray[DTYPE_float_t, ndim=2] K = np.zeros((w, h))

    i,j = np.meshgrid(I, J)
    for p in range(h):
        for q in range(w):
            K[p,q] = f( np.array([i[p,q], j[p,q]]) )

    if plot is None:
        from matplotlib import pyplot as plot

    CS = plot.contour(I, J, K, *args, **kwargs)
    plot.clabel(CS)
    plot.contour(I, J, K, (0,), linewidths=2, **kwargs)

    return plot

cdef class DecisionFunction:
    cdef np.ndarray alpha, sv
    cdef double b
    cdef int nSV
    cdef object kernel

    def __init__(self, kernel, clf, X, label):
        self.alpha = -clf.dual_coef_[0]
        #print "precomputed coef\n", alpha
        self.sv = X[clf.support_, :]
        self.nSV = len(self.sv)
        self.kernel = kernel

        clf.support_[ np.abs(self.alpha[:]) < 1. ]
        mIdx = clf.support_[ np.abs(self.alpha[:]) < 1. ]
        #print mIdx

        self.b = 0.
        for i in mIdx:
            self.b += label[i] - np.sum([ self.alpha[j]*kernel.val(X[i],self.sv[j]) for j in range(self.nSV) ])
        self.b = self.b / len(mIdx)
        #print "precomputed contant: ", b

    cpdef double eval(self, np.ndarray[DTYPE_float_t, ndim=1] x):
        cdef np.ndarray[DTYPE_float_t, ndim=1] alpha = self.alpha
        cdef np.ndarray[DTYPE_float_t, ndim=2] sv = self.sv
        cdef int i

        return np.sum([ alpha[i]*self.kernel.val(x,sv[i]) for i in range(self.nSV) ]) + self.b

def lPCA(data, d):
    N = len(data)

    # calculate the mean point of data
    m = np.average(data, axis=0)

    # ready covariance matrix
    sigma = np.zeros( (data.shape[1], data.shape[1]) )
    for x in data:
        sigma += np.outer( (x-m), (x-m) )
    sigma = 1./N * sigma

    # calculate eigen value and vector
    lmd, A = np.linalg.eig(sigma)

    # sort eigen values(vectors)
    idx = lmd.argsort()[::-1]
    lmd = lmd[idx]
    A = A[:,idx]

    # extract d basis vector(s) as basis matrix(N*d)
    u = A[:,:d]

    # mapping
    mapped = np.dot(u.T, data.T)

    # return unit vector(s) and data mapped to lower dimention space
    return (mapped.T, lmd, A)

def kPCA(data, d, kernel=None):
    # using linear lernel when not given kernel
    if kernel is None:
        if data.dtype == np.int:
            kernel = IntLinearKernel()
        if data.dtype == np.float:
            kernel = FloatLinearKernel()

    # create gram matric
    gm = kernel.gram(data)

    # calculate [J][GM]
    N = len(data)
    j = np.identity(N) - (1.0/N)* np.ones((N,N))
    jgm = np.dot(j, gm)

    # calculate eigen value and vector
    lmd, um = np.linalg.eig(jgm)

    # extract d basis vector(s) as basis matrix(N*d)
    bm = um[:,:d]

    # mapping
    mapped = np.dot(bm.T, gm)

    # return data mapped to lower dimention space
    return mapped.T

def createLabeledDataset(np.ndarray[DTYPE_int_t, ndim=2] labeled, np.ndarray[DTYPE_int_t, ndim=2] unlabeled, int label_idx=0):
    cdef float nLabeled = float(labeled.shape[0])
    cdef float nUnlabeled = float(unlabeled.shape[0])
    cdef float ratio = nUnlabeled / nLabeled

    np.random.shuffle(labeled)
    cdef np.ndarray[DTYPE_int_t, ndim=2] posdata = labeled[labeled[:,label_idx]==1,:]
    cdef np.ndarray[DTYPE_int_t, ndim=2] negdata = labeled[labeled[:,label_idx]==0,:]
    cdef float nPosdata = float(posdata.shape[0])
    cdef float nNegdata = float(negdata.shape[0])

    # HOW TO CALC. nMetaLabeled
    #------------------------------
    # nLabeled      | nUnlabeled (= nLabeled * ratio)
    # nMetaLabeled  | nMetaUnlabeled (= nMetalabeled * ratio)
    #------------------------------
    # nMetalabeled + nMetaUnlabeled = nLabeled
    # nMetalabeled + (nMetaLabeled * ratio) = nLabeled
    # nMetalabeled = nLabeled / (1 + ratio)

    cdef int nMetaLabeled = int( nLabeled / (1. + ratio) )
    cdef float scale = float(nMetaLabeled) / float(nLabeled)
    cdef int nMetaPos = int( nPosdata * scale )
    cdef int nMetaNeg = int( nNegdata * scale )

    cdef np.ndarray[DTYPE_int_t, ndim=2] metaLabeled = np.vstack( (posdata[:nMetaPos,:], negdata[:nMetaNeg,:]) )
    cdef np.ndarray[DTYPE_int_t, ndim=2] metaUnlabeled = np.vstack( (posdata[nMetaPos:,:], negdata[nMetaNeg:,:]) )

    return (metaLabeled, metaUnlabeled)

def randomSwapOverSampling(np.ndarray[DTYPE_int_t, ndim=2] X, int gain_ratio=1, int nSwap=1):
    cdef int N = len(X)
    cdef int dim = len(X[0])
    cdef int i, j, idx
    cdef np.ndarray[DTYPE_int_t, ndim=2] gained, created
    cdef np.ndarray[DTYPE_int_t, ndim=1] indices
    cdef bint isFirst = True

    for i in range(gain_ratio):
        # copy original data
        created = X.copy()

        # shuffle ndarray given as argument
        np.random.shuffle(created)

        # create new data
        for j in range(N):
            indices = np.random.randint(0, dim, nSwap)
            for idx in indices:
                created[j][idx] = X[np.random.randint(N)][idx]

        # add created data
        if isFirst:
            gained = created
            isFirst = False
        else:
            gained = np.vstack((gained, created))

    return gained

def dividingUnderSampling(np.ndarray[DTYPE_int_t, ndim=2] major, np.ndarray[DTYPE_int_t, ndim=2] minor, int ratio=1):
    cdef list trainset = []
    cdef int i, idx, width
    cdef int nMajor = major.shape[0]
    cdef int nMinor = minor.shape[0]
    cdef int nDivide = (nMajor / nMinor) / ratio

    # validation arguments
    if not nDivide > 0:
        raise ValueError('Requied two arguments, the former\'s length is larger than later\'s')
    if major.shape[1] is not minor.shape[1]:
        raise ValueError('Requied two arguments, those size is the same')

    # divide and concatenate, and create train
    np.random.shuffle(major)
    width = nMinor * ratio
    for i in range(nDivide):
        idx = i * width
        if i < nDivide - 1:
            trainset.append( np.vstack( (minor, major[idx:idx+width,:]) ) )
        else:
            trainset.append( np.vstack( (minor, major[idx:,:]) ) )

    return trainset
