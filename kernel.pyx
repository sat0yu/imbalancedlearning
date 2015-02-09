#coding: utf-8
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod
import sys

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

cdef class IntKernel:
    #__metaclass__ = ABCMeta

    #@abstractmethod
    cpdef int val(self, np.ndarray[DTYPE_int_t, ndim=1] X1, np.ndarray[DTYPE_int_t, ndim=1] X2): pass

    cpdef np.ndarray gram(self, np.ndarray[DTYPE_int_t, ndim=2] X):
        cdef int N = len(X)
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(i, N):
                gm[j,i] = gm[i,j] = self.val(X[i], X[j])
        return gm

    cpdef np.ndarray matrix(self, np.ndarray[DTYPE_int_t, ndim=2] X1, np.ndarray[DTYPE_int_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i,j] = self.val(X1[i], X2[j])
        return mat

cdef class FloatKernel:
    #__metaclass__ = ABCMeta

    #@abstractmethod
    cpdef double val(self, np.ndarray[DTYPE_float_t, ndim=1] X1, np.ndarray[DTYPE_float_t, ndim=1] X2): pass

    cpdef np.ndarray gram(self, np.ndarray[DTYPE_float_t, ndim=2] X):
        cdef int N = len(X)
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(i, N):
                gm[j,i] = gm[i,j] = self.val(X[i], X[j])
        return gm

    cpdef np.ndarray matrix(self, np.ndarray[DTYPE_float_t, ndim=2] X1, np.ndarray[DTYPE_float_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i,j] = self.val(X1[i], X2[j])
        return mat

cdef class FloatLinearKernel(FloatKernel):
    cpdef double val(self, np.ndarray[DTYPE_float_t, ndim=1] x, np.ndarray[DTYPE_float_t, ndim=1] y):
        return np.dot(x,y)

cdef class IntLinearKernel(IntKernel):
    cpdef int val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        return np.dot(x,y)

cdef class HashFunction:
    cdef int n
    cdef long [:] fuctorial
    def __init__(self, int n=2):
        assert n > 1
        self.n = n
        self.fuctorial = np.zeros(self.n, dtype=np.int)
        cdef int i
        self.fuctorial[0] = 1
        # to hash given vector efficiently,
        # containing the table storing (n+1)-power numbers
        for i in range(1,self.n):
            self.fuctorial[i] = self.fuctorial[i-1] * self.n

    cpdef long val(self, np.ndarray[DTYPE_int_t, ndim=1] x):
        assert x.shape[0] == self.n
        cdef long ret = 0
        for i in range(self.n):
            ret += self.fuctorial[i] * x[i]
        return ret

cdef class OPNgramKernel(IntKernel):
    cdef int n
    cdef object hashFunc
    def __init__(self, int n=2):
        assert n > 1
        self.n = n
        self.hashFunc = HashFunction(n)

    cpdef int val(self, np.ndarray[DTYPE_int_t, ndim=1] S, np.ndarray[DTYPE_int_t, ndim=1] T):
        cdef np.ndarray[DTYPE_int_t, ndim=1] ngram = np.zeros(self.n, dtype=np.int)
        cdef long hashed
        cdef int i, j, x, _n, lenS, lenT, rank, ret
        freq = {}

        # initialize
        _n = self.n
        lenS = S.shape[0]
        lenT = T.shape[0]
        ret = 0

        #-------------------------
        # coding substrings in S
        # and store the count of its appearing
        #-------------------------
        # encode the first n-gram
        substr = sorted( S[:_n].copy() )
        rank_map = {}
        for i,x in enumerate(substr): rank_map[x] = i
        for i,x in enumerate(substr): ngram[i] = rank_map[x]
        freq[self.hashFunc.val(ngram)] = 1

        # update-encode for each successive n-gram
        for i in range(1,lenS-_n+1):
            rank = 0
            for j in range(_n-1):
                ngram[j] = ngram[j+1]
                # update
                if S[i-1] > S[i+j] and S[i+j] >= S[i+_n-1]:
                    ngram[j] += 1
                elif S[i-1] <= S[i+j] and S[i+j] < S[i+_n-1]:
                    ngram[j] -= 1
                # evaluate the rank of the tail value
                if S[i+j] <= S[i+_n-1]: rank += 1

            ngram[_n-1] = rank
            hashed = self.hashFunc.val(ngram)
            if hashed in freq:
                freq[hashed] += 1
            else:
                freq[hashed] = 1
        # DEBUG
        # print(freq)

        #-------------------------
        # coding substrings in T
        # and add the count of the appearing of the op-coded substr. in S
        #-------------------------
        # encode the first n-gram
        substr = sorted( T[:_n].copy() )
        rank_map = {}
        for i,x in enumerate(substr): rank_map[x] = i
        for i,x in enumerate(substr): ngram[i] = rank_map[x]
        hashed = self.hashFunc.val(ngram)
        if hashed in freq: ret += freq[hashed]

        # update-encode for each successive n-gram
        for i in range(1,lenT-_n+1):
            rank = 0
            for j in range(_n-1):
                ngram[j] = ngram[j+1]
                # update
                if T[i-1] > T[i+j] and T[i+j] >= T[i+_n-1]:
                    ngram[j] += 1
                elif T[i-1] <= T[i+j] and T[i+j] < T[i+_n-1]:
                    ngram[j] -= 1
                # evaluate the rank of the tail value
                if T[i+j] <= T[i+_n-1]: rank += 1

            ngram[_n-1] = rank

            hashed = self.hashFunc.val(ngram)
            if hashed in freq: ret += freq[hashed]

        return ret

cdef class NormalizedOPNgramKernel:
    cdef object k

    def __init__(self, int n):
        self.k = OPNgramKernel(n)

    cpdef np.ndarray gram(self, np.ndarray[DTYPE_int_t, ndim=2] X):
        cdef int N = len(X)
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(i, N):
                gm[j,i] = gm[i,j] = self.val(X[i], X[j])
        return gm

    cpdef np.ndarray matrix(self, np.ndarray[DTYPE_int_t, ndim=2] X1, np.ndarray[DTYPE_int_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i,j] = self.val(X1[i], X2[j])
        return mat

    cpdef double val(self, np.ndarray[DTYPE_int_t, ndim=1] S, np.ndarray[DTYPE_int_t, ndim=1] T):
        cdef double denominator =  np.sqrt(self.k.val(S,S)) * np.sqrt(self.k.val(T,T))
        return self.k.val(S, T) / denominator

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

cdef class GaussKernel(FloatKernel):
    cdef double __beta

    def __init__(self, double beta):
        self.__beta = beta

    cpdef double val(self, np.ndarray[DTYPE_float_t, ndim=1] vec1, np.ndarray[DTYPE_float_t, ndim=1] vec2):
        cdef double dist = np.linalg.norm(vec1-vec2)
        return np.exp(-self.__beta*(dist**2))

cdef class PolyKernel(FloatKernel):
    cdef int __degree
    cdef double __coef0

    def __init__(self, int degree, double coef0=0.):
        self.__degree = degree
        self.__coef0 = coef0

    cpdef double val(self, np.ndarray[DTYPE_float_t, ndim=1] vec1, np.ndarray[DTYPE_float_t, ndim=1] vec2):
        return (np.dot(vec1, vec2) + self.__coef0)**self.__degree

cdef class LaplaceKernel(FloatKernel):
    cdef double alpha

    def __init__(self, double alpha):
        self.alpha = alpha

    cpdef double val(self, np.ndarray[DTYPE_float_t, ndim=1] vec1, np.ndarray[DTYPE_float_t, ndim=1] vec2):
        return np.exp(-self.alpha * np.sum(np.abs(vec1-vec2)))

cdef class NormalizedKernel(FloatKernel):
    cdef object k

    def __init__(self, object kernel):
        self.k = kernel

    cpdef double val(self, np.ndarray[DTYPE_float_t, ndim=1] vec1, np.ndarray[DTYPE_float_t, ndim=1] vec2):
        cdef double denominator =  np.sqrt(self.k.val(vec1, vec1)) * np.sqrt(self.k.val(vec2, vec2))
        return self.k.val(vec1, vec2) / denominator

