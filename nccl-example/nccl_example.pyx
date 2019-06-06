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

from libc.stdint cimport uintptr_t

import cudf

cdef extern from "nccl.h":

    cdef struct ncclComm:
        pass

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



cdef extern from "nccl_example_c.h" namespace "NCCLExample":
    
    cdef cppclass NcclClique:
        int get_clique_size()
        int get_rank()
        bool test_all_reduce()
        bool perform_reduce_on_partition(float * inp, int M, int N, int root_rank, float *output)

    NcclClique *create_clique(ncclComm_t comm, int workerId, int nWorkers)
    void get_unique_id(char *uid)

    void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId)

def unique_id():
    cdef char *uid = <char *> malloc(128 * sizeof(char))
    get_unique_id(uid)
    c_str = uid[:127]
    free(uid)
    return c_str


cdef class nccl:

    cdef ncclComm_t *comm

    cdef int size
    cdef int rank

    def __cinit__(self):
        self.comm = <ncclComm_t*>malloc(sizeof(ncclComm_t))

    def __dealloc__(self):

        comm_ = <ncclComm_t*>self.comm

        if comm_ != NULL:
            free(comm_)
            comm_ = NULL

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

            free(comm_)

    def abort(self):

        comm_ = <ncclComm_t*>self.comm
        cdef ncclResult_t result
        if comm_ != NULL:
            result = ncclCommAbort(deref(comm_))

            if result != ncclSuccess:
                err_str = ncclGetErrorString(result)
                print("NCCL_ERROR: %s" % err_str)
            free(comm_)


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



cdef class Demo_Algo:
    
    cdef NcclClique *world
    cdef int workerId
    cdef int nWorkers

    def __cinit__(self, workerId, nWorkers):
        self.workerId = workerId
        self.nWorkers = nWorkers
        self.world = NULL


    def create_clique(self, comm):

        cdef size_t temp_comm = <size_t>comm

        comm_ = <ncclComm_t*>temp_comm

        if self.world is not NULL:
            del self.world
            self.world = NULL
        else:
            self.world = create_clique(deref(comm_), self.workerId, self.nWorkers)

    def get_clique_size(self):
        if self.world == NULL:
            print("Must initialize before getting size")
        else:
            return self.world.get_clique_size()

    def get_rank(self):
        if self.world == NULL:
            print("Must initialize before getting size")
        else:
            return self.world.get_rank()

    def test_all_reduce(self):
        if self.world == NULL:
            print("Must initialize before getting size")
        else:
           return self.world.test_all_reduce();

    def test_on_partition(self, df, root_rank, out_gpu_mat):

        cdef object X_m = df.as_gpu_matrix()
        cdef uintptr_t X_ctype = X_m.device_ctypes_pointer.value
        
        cdef uintptr_t out_ctype = out_gpu_mat.device_ctypes_pointer.value
        
        cdef int m = X_m.shape[0]
        cdef int n = X_m.shape[1]

        return self.world.perform_reduce_on_partition(<float*>X_ctype,
                                        <int>m,
                                        <int>n,
                                        <int>root_rank,
                                        <float*>out_ctype)

    def __del__(self):
        del self.world



