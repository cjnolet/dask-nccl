# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import dask.distributed
import re
import os


# Current set of assumptions:
#  1. single node
#  2. ports starting from 9000 to 9000+nWorkers is not already taken
#  3. only once dask -> mpi hand-off per dask session is supported
#  4. no elasticity
#  5. OpenMPI (and for GPU apps, CUDA-aware ompi)

cdef extern from "nccl_example_c.h" namespace "NCCLExample":
    
    cdef cppclass NcclWorldBuilder:
        NcclWorldBuilder(int wid, int num)
        void init()
        void destroy()
        int get_clique_size()
        int get_rank()
        int get_device()

    NcclWorldBuilder *create_builder(int workerId, int nWorkers, char *uid)
    char* get_unique_id()
    
cdef class NCCL_World:
    
    cdef NcclWorldBuilder *world
    cdef int workerId
    cdef int nWorkers
    cdef int init_called
    
    def __cinit__(self, workerId, nWorkers):
        self.workerId = workerId
        self.nWorkers = nWorkers
        self.init_called = 0


    def create_builder(self, uniqueId):
        cdef char * uid = uniqueId
        if self.init_called != 0:
            print("Cannot initialize more than once")
        else:

            self.world = create_builder(self.workerId, self.nWorkers, uid)
            self.world.init()
            self.init_called = 1

    def get_clique_size(self):
        if self.init_called == 0:
            print("Must initialize before getting size")
        else:
            return self.world.get_clique_size()

    def get_rank(self):
        if self.init_called == 0:
            print("Must initialize before getting size")
        else:
            return self.world.get_rank()

    def get_device(self):
        if self.init_called == 0:
            print("Must initialize before getting size")
        else:
            return self.world.get_device()

    def unique_id(self):
        uid = get_unique_id()
        return uid




    def __del__(self):
        if self.init_called == 1:
            self.world.destroy()
