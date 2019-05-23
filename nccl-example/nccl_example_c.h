#include <nccl.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <stdexcept>

#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>

#define THROW(fmt, ...)                                         \
    do {                                                        \
        std::string msg;                                        \
        char errMsg[2048];                                      \
        sprintf(errMsg, "Exception occured! file=%s line=%d: ", \
                __FILE__, __LINE__);                            \
        msg += errMsg;                                          \
        sprintf(errMsg, fmt, ##__VA_ARGS__);                    \
        msg += errMsg;                                          \
        throw std::runtime_error(msg);                          \
    } while(0)

#define ASSERT(check, fmt, ...)                  \
    do {                                         \
        if(!(check))  THROW(fmt, ##__VA_ARGS__); \
    } while(0)

#define COMM_CHECK(call)                                        \
  do {                                                          \
    auto status = call;                                         \
    ASSERT(status == 0, "FAIL: call='%s'!", #call);             \
  } while (0)


#define CHECK_NCCL(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#pragma once

namespace NCCLExample {

class NcclWorldBuilder {
    
public:


    /**
     * @brief ctor
     * @param wid worker Id as received from dask world.
     * @param numWorkers number of workers
     */
    NcclWorldBuilder(int wid, int numWorkers, ncclUniqueId id):
        workerId(wid), nWorkers(numWorkers), uniqueId(id) {
                printf("Creating world builder with uniqueId=%s\n", id.internal);

        }

        /** dtor */
    ~NcclWorldBuilder() {
        if(workerId == 0) {
            printf("worker=%d: server closing port\n", workerId);
//            COMM_CHECK(MPI_Close_port(portName.c_str()));
        }
    }

    void init() {

      CHECK_NCCL(ncclCommInitRank(&intracomm, nWorkers, uniqueId, workerId));

      printf("Init called on worker=%d\n", workerId);
    }

    void destroy() {
      CHECK_NCCL(ncclCommDestroy(intracomm));
    }

    int get_clique_size() {

      printf("Getting clique size on worker=%d\n", workerId);

      int count;
      CHECK_NCCL(ncclCommCount(intracomm, &count));
      return count;
    }

    int get_rank() {
      int rank;
      CHECK_NCCL(ncclCommUserRank(intracomm, &rank));
      return rank;
    }

    int get_device() {
      int device;
      CHECK_NCCL(ncclCommCuDevice(intracomm, &device));
      return device;
    }

    void test_all_reduce() {

      int size = 32*1024*1024;

      float *sendbuf, *recvbuff;
      cudaStream_t s;

      cudaStreamCreate(&s);

      cudaMalloc(&sendbuf, size * sizeof(float));
      cudaMalloc(&recvbuff, size * sizeof(float));

      CHECK_NCCL(ncclAllReduce((const void*)sendbuf, (void*)recvbuff, size, ncclFloat, ncclSum, intracomm, s));

      cudaStreamSynchronize(s);

      printf("Allreduce completed successfully on %d\n", get_rank());

      cudaFree(sendbuf);
      cudaFree(recvbuff);
    }

private:

    /** comm handle for all the connected processes so far */
    ncclComm_t intracomm;
    ncclUniqueId uniqueId;


    /** current dask worker ID received from python world */
    int workerId;
    /** number of workers */
    int nWorkers;
};

NcclWorldBuilder *create_builder(int workerId, int nWorkers, const char *uid);

const char *get_unique_id();

}; // end namespace HelloMPI
