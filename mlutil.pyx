#coding: utf-8;
import numpy as np
cimport numpy as np

from kernel import *

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

def draw_contour(f, coodinates, plot=None, density=1., **kwargs):
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

    CS = plot.contour(I, J, K, **kwargs)
    plot.clabel(CS)
    plot.contour(I, J, K, (0,), linewidths=2)

    return plot

def create_dicision_function(kernel, clf, X, label):
    alpha = -clf.dual_coef_[0]
    print "precomputed coef\n", alpha
    sv = X[clf.support_, :]
    mIdx = clf.support_[ np.abs(alpha[:]) < 1. ]
    print mIdx

    b = 0.
    for i in mIdx:
        b += label[i] - np.sum([ alpha[j]*kernel.val(X[i],sv[j]) for j in range(len(sv)) ])
    b = b / len(mIdx)
    print "precomputed contant: ", b

    def df(np.ndarray[DTYPE_float_t, ndim=1] x):
        return np.sum([ alpha[i]*kernel.val(x,sv[i]) for i in range(len(sv)) ]) + b

    return df

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
