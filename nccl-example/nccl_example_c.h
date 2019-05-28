#include "cuda_runtime.h"

#include <iostream>

#include <cuML.hpp>
#include <cuML_comms.hpp>

#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>

#include <nccl.h>

#include <execinfo.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdexcept>

#include <unistd.h>

#include "util.h"

/** base exception class for the cuML or ml-prims project */
class Exception : public std::exception {
public:
  /** default ctor */
  Exception() throw(): std::exception(), msg() {}

  /** copy ctor */
  Exception(const Exception& src) throw(): std::exception(), msg(src.what()) {
    collectCallStack();
  }

  /** ctor from an input message */
  Exception(const std::string& _msg) throw(): std::exception(), msg(_msg) {
    collectCallStack();
  }

  /** dtor */
  virtual ~Exception() throw() {}

  /** get the message associated with this exception */
  virtual const char* what() const throw() { return msg.c_str(); }

private:
  /** message associated with this exception */
  std::string msg;

  /** append call stack info to this exception's message for ease of debug */
  // Courtesy: https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
  void collectCallStack() throw() {
#ifdef __GNUC__
    const int MaxStackDepth = 64;
    void* stack[MaxStackDepth];
    auto depth = backtrace(stack, MaxStackDepth);
    std::ostringstream oss;
    oss << std::endl << "Obtained " << depth << " stack frames" << std::endl;
    char** strings = backtrace_symbols(stack, depth);
    if (strings == nullptr) {
      oss << "But no stack trace could be found!" << std::endl;
      msg += oss.str();
      return;
    }
    ///@todo: support for demangling of C++ symbol names
    for (int i = 0; i < depth; ++i) {
      oss << "#" << i << " in " << strings[i] << std::endl;
    }
    free(strings);
    msg += oss.str();
#endif // __GNUC__
  }
};

/** macro to throw a runtime error */
#define THROW(fmt, ...)                                                        \
  do {                                                                         \
    std::string msg;                                                           \
    char errMsg[2048];                                                         \
    std::sprintf(errMsg, "Exception occured! file=%s line=%d: ", __FILE__,     \
                 __LINE__);                                                    \
    msg += errMsg;                                                             \
    std::sprintf(errMsg, fmt, ##__VA_ARGS__);                                  \
    msg += errMsg;                                                             \
    throw Exception(msg);                                            \
  } while (0)

/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)                                                \
  do {                                                                         \
    if (!(check))                                                              \
      THROW(fmt, ##__VA_ARGS__);                                               \
  } while (0)

/** check for cuda runtime API errors and assert accordingly */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    ASSERT(status == cudaSuccess, "FAIL: call='%s'. Reason:%s\n", #call,       \
           cudaGetErrorString(status));                                        \
  } while (0)


#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#pragma once

namespace NCCLExample {


/**
 * @brief This class encapsulates the logic for establishing and managing the
 * lifecycle of a NCCL communicator for a single clique. The reason for this
 * separate class is so that we can wrap as little as possible through Cython,
 * but still be able to act directly on the NCCL communicator.
 *
 * It "should" be possible to have multiple cliques running on the same worker
 * process, though this still needs to be verified through experimentation.
 */
class NcclClique {
    
public:

    /**
     * @param wid the worker id
     * @param numWorkers the number of workers in the clique
     * @param id the nccl unique id for the clique
     */
    NcclClique(ML::cumlHandle *handle, int wid, int numWorkers, ncclUniqueId id):
        workerId(wid), nWorkers(numWorkers), uniqueId(id), handle(handle) {
            printf("Creating world builder with uniqueId=%s\n", id.internal);

        communicator = &handle->getImpl().getCommunicator();
    }

    ~NcclClique() {
        if(workerId == 0) {
            printf("worker=%d: server closing port\n", workerId);
            delete handle;
        }
    }

    /**
     * @brief returns the number of ranks in the current clique
     */
    int get_clique_size() const {

      int count  = communicator->getSize();

      if(count == nWorkers)
          printf("Clique size on worker=%d successfully verified to be %d\n", workerId, nWorkers);
      else
          printf("Clique construction was not successful. Size on worker=%d verified to be %d, but should have been %d\n", workerId, count, nWorkers);


      return count;
    }


    /**
     * @brief returns the rank of the current worker in the clique
     */
    int get_rank() const {
      int rank = communicator->getRank();
      return rank;
    }

    const ML::cumlHandle *get_handle() const {
        return handle;
    }


    /**
     * @brief a simple validation that we can perform a collective
     * communication operation across the clique of workers.
     *
     * This specific example creates a float array of 10 elements,
     * all initialized to 1, and performs an allReducem to sum the
     * corresponding elements of each of the arrays together.
     */
    bool test_all_reduce() {

      printf("Calling allReduce on %d\n", workerId);

      int size = 10;
      int num_bytes = size * sizeof(float);

      float *sendbuf, *recvbuff;
      cudaStream_t s;

      CUDA_CHECK(cudaStreamCreate(&s));

      CUDA_CHECK(cudaMalloc((void**)&sendbuf, num_bytes));
      CUDA_CHECK(cudaMalloc((void**)&recvbuff, num_bytes));

      init_dev_arr<float>(sendbuf, size, 1.0f, s);
      init_dev_arr<float>(recvbuff, size, 0.0f, s);

      print(sendbuf, size, "sent", s);

      communicator->allreduce((const void*)sendbuf, (void*)recvbuff, size, MLCommon::cumlCommunicator::FLOAT, MLCommon::cumlCommunicator::SUM, s);

      CUDA_CHECK(cudaStreamSynchronize(s));

      print(recvbuff, size, "received", s);

      bool verify = verify_dev_arr(recvbuff, size, (float)nWorkers, s);
      if(verify)
          printf("allReduce completed successfully on %d. Received values verified to be %d\n", workerId, nWorkers);
      else
          printf("allReduce did not contain the expected values [%d] on %d\n", nWorkers, workerId);


      CUDA_CHECK(cudaFree(sendbuf));
      CUDA_CHECK(cudaFree(recvbuff));

      return verify;

    }

    bool perform_reduce_on_partition(float *input, int M, int N, int root_rank) {

        int n_workers = nWorkers;
        int rank = communicator->getRank();

        int num_bytes = M*N * sizeof(float);

        float *sendbuf = input, *recvbuff;
        cudaStream_t s;

        CUDA_CHECK(cudaStreamCreate(&s));

        if(rank == root_rank) {
          CUDA_CHECK(cudaMalloc((void**)&recvbuff, num_bytes));
          init_dev_arr<float>(recvbuff, M*N, 0.0f, s);
        }

        print(sendbuf, M*N, "sent", s);

        communicator->reduce((const void*)sendbuf, (void*)recvbuff, M*N*sizeof(float),
            MLCommon::cumlCommunicator::FLOAT, MLCommon::cumlCommunicator::SUM, root_rank, s);

        CUDA_CHECK(cudaStreamSynchronize(s));

        print(recvbuff, M*N, "received", s);

        bool verify = true;
        if(rank == root_rank) {
          verify_dev_arr(recvbuff, M*N, (float)n_workers, s);
          if(verify)
            printf("allReduce completed successfully on %d. Received values verified to be %d\n", rank, n_workers);
          else
            printf("allReduce did not contain the expected values [%d] on %d\n", n_workers, rank);
        }


        CUDA_CHECK(cudaFree(sendbuf));

        if(rank == root_rank)
          CUDA_CHECK(cudaFree(recvbuff));

        return verify;
    }

private:

    /** comm handle for all the connected processes so far */

    ncclUniqueId uniqueId;


    ML::cumlHandle *handle;
    const MLCommon::cumlCommunicator *communicator;

    /** current dask worker ID received from python world */
    int workerId;
    /** number of workers */
    int nWorkers;
};

NcclClique *create_clique(int workerId, int nWorkers, const char *uid);

const char *get_unique_id();



}; // end namespace HelloMPI
