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
        cdef int i, j, k=0, slen=len(s), tlen=len(t)

        if slen < self.p or tlen < self.p: return 0

        for i in range( slen - (self.p - 1) ):
            for j in range( tlen - (self.p - 1) ):
                k += 1 if s[i:i+self.p] == t[j:j+self.p] else 0

        return k

cdef class NormalizedSpectrumKernel(FloatStringKernel):
    cdef int p

    def __init__(self, int p):
        self.p = p

    cpdef double val(self, s, t):
        cdef int i,j,k=0,ss=0,tt=0,slen=len(s),tlen=len(t)

        if slen < self.p or tlen < self.p: return 0.

        for i in range( slen - (self.p - 1) ):
            for j in range( slen - (self.p - 1) ):
                ss += 1 if s[i:i+self.p] == s[j:j+self.p] else 0

        for i in range( tlen - (self.p - 1) ):
            for j in range( tlen - (self.p - 1) ):
                tt += 1 if t[i:i+self.p] == t[j:j+self.p] else 0

        for i in range( slen - (self.p - 1) ):
            for j in range( tlen - (self.p - 1) ):
                k += 1 if s[i:i+self.p] == t[j:j+self.p] else 0

        return k / np.sqrt( ss * tt )
