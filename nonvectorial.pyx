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
                gm[j][i] = gm[i][j] = self.val(X[i], X[j])
        return gm

    cpdef np.ndarray matrix(self, np.ndarray[DTYPE_int_t, ndim=2] X1, np.ndarray[DTYPE_int_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i][j] = self.val(X1[i], X2[j])
        return mat

cdef extern from "string.h":
    int strlen(char*)
    int strncmp(char*, char*, int)

cdef class SpectrumKernel:
    cdef int p

    def __init__(self, int p):
        self.p = p

    cpdef int val(self, char* s, char* t):
        cdef int i,j,k=0

        if not (strlen(s) and strlen(t)): return 0

        for i in range( strlen(s) - (self.p - 1) + 1 ):
            for j in range( strlen(t) - (self.p - 1) + 1 ):
                k += 0 if strncmp(s + i, t + j, self.p) else 1

        return k

