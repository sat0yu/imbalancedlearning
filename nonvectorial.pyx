#coding: utf-8
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod
import sys

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp, strlen, strncmp
from cpython.string cimport PyString_AsString

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

cdef class SpectrumKernel:
    cdef int p

    def __init__(self, int p):
        self.p = p

    # this method doesn't work well..., something wrong
    cdef char** to_cstring_array(self, list_str):
        cdef int N = len(list_str)
        cdef char **ret = <char **>malloc( N * sizeof(char *) )

        for i in range(N):
            ret[i] = PyString_AsString(list_str[i])
            printf("%s\n",ret[i]);

        return ret

    cpdef np.ndarray gram(self, list_str):
        cdef int i, j, N = len(list_str)
        cdef char** X = <char **>malloc( N * sizeof(char *) )
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)

        for i in range(N):
            X[i] = PyString_AsString(list_str[i])

        for i in range(N):
            for j in range(i, N):
                gm[j,i] = gm[i,j] = self.val(X[i], X[j])

        free(X)
        return gm

    cdef int val(self, char* s, char* t):
        cdef int i,j,k=0

        if not (strlen(s) and strlen(t)): return 0

        for i in range( strlen(s) - (self.p - 1) + 1 ):
            for j in range( strlen(t) - (self.p - 1) + 1 ):
                k += 0 if strncmp(s + i, t + j, self.p) else 1

        return k

