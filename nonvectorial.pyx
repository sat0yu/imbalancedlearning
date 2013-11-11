#coding: utf-8
import numpy as np
cimport numpy as np
import sys

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

cdef class IntStringKernel:
    cpdef np.ndarray gram(self, X):
        cdef int i, j, N = len(X)
        cdef np.ndarray[DTYPE_int_t, ndim=2] gm = np.identity(N, dtype=DTYPE_int)

        for i in range(N):
            for j in range(i, N):
                gm[j,i] = gm[i,j] = self.val(X[i], X[j])

        return gm

    cpdef np.ndarray matrix(self, X1, X2):
        cdef int i, j, N=len(X1), M=len(X2)
        cdef np.ndarray[DTYPE_int_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_int)

        for i in range(N):
            for j in range(M):
                mat[i,j] = self.val(X1[i], X2[j])

        return mat

cdef class FloatStringKernel:
    cpdef np.ndarray gram(self, X):
        cdef int i, j, N = len(X)
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)

        for i in range(N):
            for j in range(i, N):
                gm[j,i] = gm[i,j] = self.val(X[i], X[j])

        return gm

    cpdef np.ndarray matrix(self, X1, X2):
        cdef int i, j, N=len(X1), M=len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)

        for i in range(N):
            for j in range(M):
                mat[i,j] = self.val(X1[i], X2[j])

        return mat

cdef class SpectrumKernel(IntStringKernel):
    cdef int p

    def __init__(self, int p):
        self.p = p

    cpdef int val(self, s, t):
        cdef int i, k=0, slen=len(s), tlen=len(t)
        cdef dict buf = {}

        if slen < self.p or tlen < self.p: return 0

        for i in range( slen - (self.p - 1) ):
            try:
                buf[ s[i:i+self.p] ] += 1
            except KeyError:
                buf[ s[i:i+self.p] ] = 1

        for i in range( tlen - (self.p - 1) ):
            try:
                k += buf[ t[i:i+self.p] ]
            except KeyError:
                pass

        return k

cdef class NormalizedSpectrumKernel(FloatStringKernel):
    cdef int p

    def __init__(self, int p):
        self.p = p

    cpdef int sub_val(self, s, t):
        cdef int i, k=0, slen=len(s), tlen=len(t)
        cdef dict buf = {}

        for i in range( slen - (self.p - 1) ):
            try:
                buf[ s[i:i+self.p] ] += 1
            except KeyError:
                buf[ s[i:i+self.p] ] = 1

        for i in range( tlen - (self.p - 1) ):
            try:
                k += buf[ t[i:i+self.p] ]
            except KeyError:
                pass

        return k


    cpdef double val(self, s, t):
        if len(s) < self.p or len(t) < self.p: return 0.

        return self.sub_val(s,t) / np.sqrt( self.sub_val(s,s) * self.sub_val(t,t) )
