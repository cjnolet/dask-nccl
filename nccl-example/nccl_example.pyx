# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import dask.distributed
from libcpp cimport bool
import re
import os


cdef extern from "nccl_example_c.h" namespace "NCCLExample":
    
    cdef cppclass NcclClique:
        int get_clique_size()
        int get_rank()
        bool test_all_reduce()

    NcclClique *create_clique(int workerId, int nWorkers, char *uid)
    char* get_unique_id()


def unique_id():
    cdef const char *uid = get_unique_id()
    c_str = uid[:127]
    return c_str


cdef class NCCL_Clique:
    
    cdef NcclClique *world
    cdef int workerId
    cdef int nWorkers

    def __cinit__(self, workerId, nWorkers):
        self.workerId = workerId
        self.nWorkers = nWorkers
        self.world = NULL


    def create_clique(self, uniqueId):

        print(len(uniqueId))
        cdef char * uid = uniqueId
        if self.world is not NULL:
            del self.world
            self.world = NULL
        else:
            self.world = create_clique(self.workerId, self.nWorkers, uid)

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

    def __del__(self):
        del self.world
