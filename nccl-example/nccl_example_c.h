#include "cuda_runtime.h"

#include <iostream>

#include <nccl.h>

#include <execinfo.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdexcept>

#include <unistd.h>

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
    NcclClique(int wid, int numWorkers, ncclUniqueId id):
        workerId(wid), nWorkers(numWorkers), uniqueId(id) {
                printf("Creating world builder with uniqueId=%s\n", id.internal);
                init();
        }

    ~NcclClique() {
        if(workerId == 0) {
            printf("worker=%d: server closing port\n", workerId);
            destroy();
        }
    }

    /**
     * @brief returns the number of ranks in the current clique
     */
    int get_clique_size() {

      int count;
      NCCL_CHECK(ncclCommCount(comm, &count));

      if(count == nWorkers)
          printf("Clique size on worker=%d successfully verified to be %d\n", workerId, nWorkers);
      else
          printf("Clique construction was not successful. Size on worker=%d verified to be %d, but should have been %d\n", workerId, count, nWorkers);


      return count;
    }


    /**
     * @brief returns the rank of the current worker in the clique
     */
    int get_rank() {
      int rank;
      NCCL_CHECK(ncclCommUserRank(comm, &rank));
      return rank;
    }

    /**
     * @brief returns the GPU device number of the current worker
     * in the clique. Note that when this is used with Dask and
     * the LocalCUDACluster, it should always return 0, as that
     * will be the first device ordinal from the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    int get_device() {
      int device;
      NCCL_CHECK(ncclCommCuDevice(comm, &device));
      return device;
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

      NCCL_CHECK(ncclAllReduce((const void*)sendbuf, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));

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

    /**
     * @brief simple utility function to print an array of floats
     * that lives on the host.
     */
    template<typename T>
    void print(T *arr, int size, std::string name, cudaStream_t s) {

    float *res = (T*)malloc(size*sizeof(T));
    CUDA_CHECK(cudaMemcpyAsync(res, arr, size*sizeof(T), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::cout << name << " = [";
    for(int i = 0; i < size; i++) {
        std::cout << res[i] << " ";

        if(i < size-1)
            std::cout << ", ";
        }

        std::cout << "]" << std::endl;
        free(res);
    }

    /**
     * @brief simple utility function to initialize all the items in a device
     * array to the given value.
     */
    template<typename T>
    void init_dev_arr(T *devArr, int size, T value, cudaStream_t s) {
        T *h_init = (T*)malloc(size * sizeof(T));
        for(int i = 0; i < size; i++)
            h_init[i] = value;
        CUDA_CHECK(cudaMemcpyAsync(devArr, h_init, size*sizeof(T), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        free(h_init);
    }

    /**
     * @brief simple utility function to verify all the items in a device
     * array equal the given value.
     */
    template<typename T>
    bool verify_dev_arr(T *devArr, int size, T value, cudaStream_t s) {

        bool ret = true;

        T *h_init = (T*)malloc(size * sizeof(T));
        CUDA_CHECK(cudaMemcpyAsync(h_init, devArr, size*sizeof(T), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        for(int i = 0; i < size; i++)
            if(h_init[i] != value)
                ret = false;

        free(h_init);

        return ret;
    }

private:

    /** comm handle for all the connected processes so far */
    ncclComm_t comm;
    ncclUniqueId uniqueId;

    /** current dask worker ID received from python world */
    int workerId;
    /** number of workers */
    int nWorkers;

    /**
     * @brief Initializes the communicator for the nccl clique
     */
    void init() {
      NCCL_CHECK(ncclCommInitRank(&comm, nWorkers, uniqueId, workerId));
      printf("Init called on worker=%d\n", workerId);
    }

    /**
     * @brief destroys the communicator for the nccl clique
     */
    void destroy() {
      NCCL_CHECK(ncclCommDestroy(comm));
    }
};

NcclClique *create_clique(int workerId, int nWorkers, const char *uid);

const char *get_unique_id();

}; // end namespace HelloMPI
