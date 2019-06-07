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

const MLCommon::cumlCommunicator *build_comm(ncclComm_t comm, int workerId, int nWorkers);

int get_clique_size(const MLCommon::cumlCommunicator *communicator);

int get_rank(const MLCommon::cumlCommunicator *communicator);

bool fit(const MLCommon::cumlCommunicator *communicator,
    int nWorkers, float *sendbuf, int M, int N, int root_rank, float *recvbuff);

void get_unique_id(char *uid);

void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId);

}; // end namespace NCCLExample
