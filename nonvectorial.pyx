#coding: utf-8
import numpy as np
cimport numpy as np
import sys

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp, strlen, strncmp
from cpython.string cimport PyString_AsString

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

cdef class StringKernel:
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

    cpdef np.ndarray matrix(self, list_str1, list_str2):
        cdef int i, j, N=len(list_str1), M=len(list_str2)
        cdef char** X1 = <char **>malloc( N * sizeof(char *) )
        cdef char** X2 = <char **>malloc( M * sizeof(char *) )
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)

        for i in range(N):
            X1[i] = PyString_AsString(list_str1[i])
        for i in range(M):
            X2[i] = PyString_AsString(list_str2[i])

        for i in range(N):
            for j in range(M):
                mat[i,j] = self.val(X1[i], X2[j])

        free(X1)
        free(X2)
        return mat

cdef class SpectrumKernel(StringKernel):
    cdef int p

    def __init__(self, int p):
        self.p = p

    cpdef int val(self, char* s, char* t):
        cdef int i,j,k=0

        if strlen(s) < self.p or strlen(t) < self.p: return 0

        for i in range( strlen(s) - (self.p - 1) + 1 ):
            for j in range( strlen(t) - (self.p - 1) + 1 ):
                k += 0 if strncmp(s + i, t + j, self.p) else 1

        return k

cdef class NormalizedSpectrumKernel(StringKernel):
    cdef int p

    def __init__(self, int p):
        self.p = p

    cpdef double val(self, char* s, char* t):
        cdef int i,j,k=0,ss=0,tt=0
        cdef double denominator

        if strlen(s) < self.p or strlen(t) < self.p: return 0.

        for i in range( strlen(s) - (self.p - 1) + 1 ):
            for j in range( strlen(s) - (self.p - 1) + 1 ):
                ss += 0 if strncmp(s + i, s + j, self.p) else 1

        for i in range( strlen(t) - (self.p - 1) + 1 ):
            for j in range( strlen(t) - (self.p - 1) + 1 ):
                tt += 0 if strncmp(t + i, t + j, self.p) else 1

        denominator = np.sqrt(ss) * np.sqrt(tt)

        for i in range( strlen(s) - (self.p - 1) + 1 ):
            for j in range( strlen(t) - (self.p - 1) + 1 ):
                k += 0 if strncmp(s + i, t + j, self.p) else 1

        return k / denominator
