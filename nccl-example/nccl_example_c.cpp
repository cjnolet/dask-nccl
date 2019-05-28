#include "nccl_example_c.h"

#include <cuML.hpp>
#include <cuML_comms.hpp>

#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>

#include <stdio.h>
#include <unistd.h>
#include <string>
#include <stdexcept>

#include <iostream>


namespace NCCLExample {

/**
 * @brief Factory function to build a NCCL clique. This function is meant to be
 * called on all the ranks that will be joining the clique concurrently.
 *
 * @param workerId the rank given to the current worker/process/thread.
 *                 This must be unique across the clique
 * @param nWorkers the number of workers that will be joining the clique
 * @param uniqueId the character array from the NCCL-generated uniqueId
 */
NcclClique *create_clique(int workerId, int nWorkers, const char *uniqueId) {

    printf("Creating clique with worker=%d\n", workerId);

    std::cout << "uniqueId in cpp: " << uniqueId << std::endl;

    ncclUniqueId id;
    memcpy(id.internal, uniqueId, NCCL_UNIQUE_ID_BYTES);

    ML::cumlHandle *handle = new ML::cumlHandle(); // in this example, the NcclClique will take ownership of handle.
    ncclComm_t comm;
    initialize_comms(*handle, comm, nWorkers, workerId, id);

    return new NcclClique(handle, workerId, nWorkers, id);
}

/**
 * @brief Returns a NCCL unique ID as a character array. PyTorch
 * uses this same approach, so that it can be more easily
 * converted to a native Python string by Cython and further
 * serialized to be sent across process & node boundaries.
 *
 * @returns the generated NCCL unique ID for establishing a
 * new clique.
 */
const char* get_unique_id() {

  ncclUniqueId id;
  ncclGetUniqueId(&id);

  // NCCL's uniqueId type is just a struct
  // with an `internal` field.
  char* newchar = id.internal;
  return newchar;
}
    
bool perform_reduce_on_partition(const ML::cumlHandle &handle, float *input, int M, int N, int n_workers, int root_rank) {
    
    const MLCommon::cumlCommunicator *communicator = &handle.getImpl().getCommunicator();
    
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



} // end namespace HelloMPI
