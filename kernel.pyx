#coding: utf-8
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod
import sys

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

class IntKernel():
    __metaclass__ = ABCMeta

    @abstractmethod
    def val(self, np.ndarray[DTYPE_int_t, ndim=2] X1, np.ndarray[DTYPE_int_t, ndim=2] X2): pass

    def gram(self, np.ndarray[DTYPE_int_t, ndim=2] X):
        cdef int N = len(X)
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(i, N):
                gm[j][i] = gm[i][j] = self.val(X[i], X[j])
        return gm

    def matrix(self, np.ndarray[DTYPE_int_t, ndim=2] X1, np.ndarray[DTYPE_int_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i][j] = self.val(X1[i], X2[j])
        return mat

class FloatKernel():
    __metaclass__ = ABCMeta

    @abstractmethod
    def val(self, np.ndarray[DTYPE_float_t, ndim=2] X1, np.ndarray[DTYPE_float_t, ndim=2] X2): pass

    def gram(self, np.ndarray[DTYPE_float_t, ndim=2] X):
        cdef int N = len(X)
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(i, N):
                gm[j][i] = gm[i][j] = self.val(X[i], X[j])
        return gm

    def matrix(self, np.ndarray[DTYPE_float_t, ndim=2] X1, np.ndarray[DTYPE_float_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i][j] = self.val(X1[i], X2[j])
        return mat

class FloatLinearKernel(FloatKernel):
    def val(self, np.ndarray[DTYPE_float_t, ndim=1] x, np.ndarray[DTYPE_float_t, ndim=1] y):
        return np.dot(x,y)

class IntLinearKernel(IntKernel):
    def val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        return np.dot(x,y)

class WeightendHammingKernel(IntKernel):
    def __init__(self, np.ndarray[DTYPE_float_t, ndim=1] _w, int _d=1):
        self.__w = _w
        self.__dim = len(_w)
        self.__d = _d

    def val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        # if not ( self.__dim == len(x) and self.__dim == len(y) ):
        #     raise ValueError('Required two arguments, those size are the same as the weight vector\'s one')

        # in numpy, == operator of ndarrays means
        # correspondings for each feature
        correspondings = (x == y)

        # in numpy, True equals 1 and False equals 0
        # so, numpy.dot() can calculate expectedly
        return (np.dot(self.__w, correspondings))**self.__d

class CustomHammingKernel(IntKernel):
    def __init__(self, _hash, int _idx, double _var=1.0, int _d=1):
        self.__hash = _hash
        self.__idx = _idx
        self.__var = _var
        self.__d = _d

    def val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        cdef int hash_x = self.__hash.get(x[self.__idx], 0)
        cdef int hash_y = self.__hash.get(y[self.__idx], 0)
        cdef double gauss = np.exp( ( -(hash_x - hash_y)**2 )/ self.__var)

        return ( gauss + float(sum( x == y )) )**self.__d

class CustomKernel(IntKernel):
    def __init__(self, int _nFeatures, double _var=1.0, int _d=1):
        self.__var = _var
        self.__d = _d
        self.__nFeatures = _nFeatures

    def val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        cdef double gauss = sum( np.exp( -np.abs(x[self.__nFeatures:] - y[self.__nFeatures:])**2 / self.__var ) )
        return gauss + sum( x[:self.__nFeatures] == y[:self.__nFeatures] )**self.__d

class HammingKernel(IntKernel):
    def __init__(self, int d=1):
        self.__d = d

    def val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        return sum( x == y )**self.__d

class GaussKernel(FloatKernel):
    def __init__(self, double beta):
        self.__beta = beta

    def val(self, np.ndarray[DTYPE_float_t, ndim=1] vec1, np.ndarray[DTYPE_float_t, ndim=1] vec2):
        cdef double dist = np.linalg.norm(vec1-vec2)
        return np.exp(-self.__beta*(dist**2))
