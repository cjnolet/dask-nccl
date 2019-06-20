# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
from cuml.common.handle import Handle

from cpython cimport array
import array

import dask.distributed
from libcpp cimport bool
from libc.stdlib cimport malloc, free
import re
import os
from cython.operator cimport dereference as deref

import numpy as np
import numba.cuda
import cudf

from libc.stdint cimport uintptr_t

import cudf

cdef extern from "nccl.h":

    cdef struct ncclComm

    ctypedef struct ncclUniqueId:
        char *internal[128]

    ctypedef ncclComm *ncclComm_t

    ctypedef enum ncclResult_t:
        ncclSuccess
        ncclUnhandledCudaError
        ncclSystemError
        ncclInternalError
        ncclInvalidArgument
        ncclInvalidUsage
        ncclNumResults

    ncclResult_t ncclCommInitRank(ncclComm_t *comm,
                                  int nranks,
                                  ncclUniqueId commId,
                                  int rank)

    ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId)

    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank)

    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *count)

    const char *ncclGetErrorString(ncclResult_t result)

    ncclResult_t ncclCommAbort(ncclComm_t comm)

    ncclResult_t ncclCommDestroy(ncclComm_t comm)



cdef extern from "common/cuML_comms_impl.cpp" namespace "MLCommon":
    cdef cppclass cumlCommunicator

cdef extern from "cuML.hpp" namespace "ML" nogil:
    cdef cppclass cumlHandle:
        cumlHandle() except +

cdef extern from "simple_reduce_api.h" namespace "NCCLExample":

    void inject_comms_py(cumlHandle *handle, ncclComm_t comm, void *ucp_worker, void *eps, int size, int rank)
    
    int get_clique_size(const cumlCommunicator * communicator)

    int get_rank(const cumlCommunicator * communicator)
    
    void test_ep(void *ep);

    bool fit(cumlHandle &handle,
                                    int nWorkers,
                                    float * sendbuf,
                                    int M,
                                    int N,
                                    int root_rank,
                                    float * recvbuff)

    void get_unique_id(char * uid)

    void ncclUniqueIdFromChar(ncclUniqueId * id, char * uniqueId);


def unique_id():
    cdef char *uid = <char *> malloc(128 * sizeof(char))
    get_unique_id(uid)
    c_str = uid[:127]
    free(uid)
    return c_str


def test_endpoint(ep):
    test_ep(<void*><size_t>ep)

def inject_comms_on_handle(handle, nccl_inst, ucp_worker, eps, size, rank):

    cdef size_t *ucp_eps = <size_t*> malloc(len(eps)*sizeof(size_t))

    cdef size_t ep_st
    for i in range(len(eps)):
        if eps[i] is not None:
            ucp_eps[i] = <size_t>eps[i].get_ep()
            test_ep(<void*><size_t>eps[i].get_ep())
        else:
            ucp_eps[i] = 0

    cdef void* ucp_worker_st = <void*><size_t>ucp_worker

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <cumlHandle*>handle_size_t

    cdef size_t nccl_comm_size_t = <size_t>nccl_inst.get_comm()
    nccl_comm_ = <ncclComm_t*>nccl_comm_size_t

    inject_comms_py(handle_, deref(nccl_comm_), <void*>ucp_worker_st, <void*>ucp_eps, size, rank)


cdef class nccl:

    cdef ncclComm_t *comm

    cdef int size
    cdef int rank

    def __cinit__(self):
        self.comm = <ncclComm_t*>malloc(sizeof(ncclComm_t))

    def __dealloc__(self):

         comm_ = <ncclComm_t*>self.comm

         if comm_ != NULL:
             free(self.comm)
             self.comm = NULL

    @staticmethod
    def get_unique_id():
        return unique_id()

    def init(self, nranks, commId, rank):

        self.size = nranks
        self.rank = rank

        cdef ncclUniqueId *ident = <ncclUniqueId*>malloc(sizeof(ncclUniqueId))
        ncclUniqueIdFromChar(ident, commId)

        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result = ncclCommInitRank(comm_, nranks, deref(ident), rank)

        if result != ncclSuccess:
            err_str = ncclGetErrorString(result)
            print("NCCL_ERROR: %s" % err_str)

    def destroy(self):

        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result
        if comm_ != NULL:
            result = ncclCommDestroy(deref(comm_))

            if result != ncclSuccess:
                err_str = ncclGetErrorString(result)
                print("NCCL_ERROR: %s" % err_str)

            free(self.comm)
            self.comm = NULL

    def abort(self):

        comm_ = <ncclComm_t*>self.comm
        cdef ncclResult_t result
        if comm_ != NULL:
            result = ncclCommAbort(deref(comm_))

            if result != ncclSuccess:
                err_str = ncclGetErrorString(result)
                print("NCCL_ERROR: %s" % err_str)
            free(comm_)
            self.comm = NULL


    def cu_device(self):
        cdef int *dev = <int*>malloc(sizeof(int))

        comm_ = <ncclComm_t*>self.comm
        cdef ncclResult_t result = ncclCommCuDevice(deref(comm_), dev)

        if result != ncclSuccess:
            err_str = ncclGetErrorString(result)
            print("NCCL_ERROR: %s" % err_str)

        ret = dev[0]
        free(dev)
        return ret

    def user_rank(self):

        cdef int *rank = <int*>malloc(sizeof(int))

        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result = ncclCommUserRank(deref(comm_), rank)

        if result != ncclSuccess:
            err_str = ncclGetErrorString(result)
            print("NCCL_ERROR: %s" % err_str)

        ret = rank[0]
        free(rank)
        return ret

    def get_comm(self):
        return <size_t>self.comm

class SimpleReduce:
    
    def __init__(self, workerId, nWorkers, handle):
        self.workerId = workerId
        self.nWorkers = nWorkers
        self.handle = handle

    def fit(self, df):
        """
        Mimics an MNMG fit() function on a model that uses collective comms
        """
        cdef object X_m = df.as_gpu_matrix()
        cdef uintptr_t X_ctype = X_m.device_ctypes_pointer.value

        if self.workerId == 0:
            out_gpu_mat = numba.cuda.to_device(np.zeros((df.shape[0], df.shape[1]),
                                                        dtype=np.float32, order="F"))
            out_df = cudf.DataFrame(index=cudf.dataframe.RangeIndex(0, df.shape[0]))
        else:
            out_gpu_mat = numba.cuda.device_array((1, 1), dtype=np.float32)
            out_df = None

        cdef uintptr_t out_ctype = out_gpu_mat.device_ctypes_pointer.value
        
        cdef int m = X_m.shape[0]
        cdef int n = X_m.shape[1]

        self.model_params = out_df
        
        cdef cumlHandle *handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        self.reduce_result = fit(handle_[0],
                                self.nWorkers,
                                <float*>X_ctype,
                                <int>m,
                                <int>n,
                                <int>0,
                                <float*>out_ctype)

        if self.workerId == 0:
            for i in range(0, out_gpu_mat.shape[1]):
                out_df[str(i)] = out_gpu_mat[:, i]


        return self

    def verify(self):
        return self.reduce_result

    def __del__(self):
        del self.cumlComm




