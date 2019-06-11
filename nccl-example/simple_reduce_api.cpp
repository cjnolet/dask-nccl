#include "simple_reduce_api.h"

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


void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId) {
    memcpy(id->internal, uniqueId, NCCL_UNIQUE_ID_BYTES);
}


/**
 * @brief Factory function to build a NCCL clique. This function is meant to be
 * called on all the ranks that will be joining the clique concurrently.
 *
 * @param workerId the rank given to the current worker/process/thread.
 *                 This must be unique across the clique
 * @param nWorkers the number of workers that will be joining the clique
 * @param uniqueId the character array from the NCCL-generated uniqueId
 */
const MLCommon::cumlCommunicator *build_comm(ncclComm_t comm, ucp_worker_h *worker, ucp_ep_h **eps, int workerId, int nWorkers) {

    printf("Creating clique with comm=%s, nWorkers=%d, worker=%d\n", comm, nWorkers, workerId);

    int rank = -1;

    NCCL_CHECK(ncclCommUserRank(comm, &rank));

    printf("Verified rank = %d\n", rank);

    ML::cumlHandle *handle = new ML::cumlHandle(); // in this example, the NcclClique will take ownership of handle.
    inject_comms(*handle, comm, nullptr, nullptr, nWorkers, workerId);

    return &handle->getImpl().getCommunicator();
}

/**
 * @brief returns the number of ranks in the current clique
 */
int get_clique_size(const MLCommon::cumlCommunicator *communicator) {
  int count  = communicator->getSize();
  return count;
}


/**
 * @brief returns the rank of the current worker in the clique
 */
int get_rank(const MLCommon::cumlCommunicator *communicator) {
  int rank = communicator->getRank();
  return rank;
}



bool fit(const MLCommon::cumlCommunicator *communicator, int nWorkers, float *sendbuf, int M, int N, int root_rank, float *recvbuff) {

    int n_workers = nWorkers;
    int rank = communicator->getRank();

    int num_bytes = M*N * sizeof(float);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    communicator->reduce((const void*)sendbuf, (void*)recvbuff, M*N,
        MLCommon::cumlCommunicator::FLOAT, MLCommon::cumlCommunicator::SUM, root_rank, s);

    CUDA_CHECK(cudaStreamSynchronize(s));

    bool verify = true;
    if(rank == root_rank) {
      verify_dev_arr(recvbuff, M*N, (float)n_workers, s);
      if(verify)
        printf("Reduce on partition completed successfully on %d. Received values verified to be %d\n", rank, n_workers);
      else
        printf("Reduce on partition did not contain the expected values [%d] on %d\n", n_workers, rank);
    }

    return verify;
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
void get_unique_id(char *uid) {

  ncclUniqueId id;
  ncclGetUniqueId(&id);

  memcpy(uid, id.internal, NCCL_UNIQUE_ID_BYTES);
}
    


} // end namespace HelloMPI
