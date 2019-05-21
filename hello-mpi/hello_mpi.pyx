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

cdef extern from "hello_mpi_c.h" namespace "HelloMPI":
    
    cdef cppclass MpiWorldBuilder
    
    MpiWorldBuilder* mpi_run(int workerId, int nWorkers)
    void mpi_init(int workerId)
    int get_rank(MpiWorldBuilder *comm, int workerId)
    void mpi_finalize(int workerId)
    
cdef class World:
    
    cdef MpiWorldBuilder *world
    cdef int workerId
    cdef int nWorkers
    cdef int init_called
    
    def __cinit__(self, workerId, nWorkers):
        self.workerId = workerId
        self.nWorkers = nWorkers
        self.init_called = 0
        
    def rank(self):
        return get_rank(self.world, self.workerId)    

    def init(self):
        if self.init_called != 0:
            print("Init has already been called!")
        else:
            self.world = mpi_run(self.workerId, self.nWorkers)
            self.init_called = 1

    def finalize(self):
        if self.init_called == 0:
            print("Cannot finalize until init has been called")
        else:
            mpi_finalize(self.workerId)
        
    def __del__(self):
        if self.init_called == 1:
            self.finalize()