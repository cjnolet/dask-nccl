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


cdef extern from "ucp/api/ucp.h" nogil:
    
    cdef struct ucp_worker:
        pass

    ctypedef ucp_worker *ucp_worker_h
    
    ctypedef struct ucp_address_t
    
    ctypedef struct ucs_status_t
    

    ucs_status_t ucp_worker_get_address	(ucp_worker_h 	worker, 
                                         ucp_address_t ** address_p, 
                                         size_t * 	address_length_p)		    

def get_address(worker):
    
    cdef ucp_worker_h *w = <ucp_worker_h*><size_t>worker
    
    cdef ucp_address_t **localAdd = <ucp_address_t**>malloc(sizeof(ucp_address_t*))
    
    cdef size_t *addr_length = <size_t*>malloc(sizeof(size_t))
    
    with nogil:
        ucp_worker_get_address(deref(w), 
                              localAdd,
                              addr_length)



