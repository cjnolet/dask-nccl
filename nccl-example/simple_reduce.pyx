# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import dask.distributed
from libcpp cimport bool
from libc.stdlib cimport malloc, free
import re
import os
from cython.operator cimport dereference as deref

import numpy as np
import numba.cuda

from libc.stdint cimport uintptr_t


cdef extern from "ucp_helper.h" nogil:

    void call_ucp_function(void* ucp_worker)
    

def ucp_function(worker):
    
    cdef void *w = <void*><size_t>worker
    
    with nogil:
        call_ucp_function(w)



