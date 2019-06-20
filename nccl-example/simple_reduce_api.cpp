#include "simple_reduce_api.h"

#include <cuML.hpp>
#include <cuML_comms.hpp>

#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>

#include <ucp/api/ucp.h>

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



bool fit(ML::cumlHandle &handle, int nWorkers, float *sendbuf, int M, int N, int root_rank, float *recvbuff) {

    const MLCommon::cumlCommunicator *communicator = &handle.getImpl().getCommunicator();
    
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
    
    
    int *to_send = new int[5] {1, 2, 3, 4, 5};
    int *to_recv = (int*)malloc(sizeof(int)*5);
    
    MLCommon::cumlCommunicator::request_t req;
    if(rank == 0) {
        printf("About to send data on 0\n");
        communicator->isend(to_send, 5*sizeof(int), 1, 50, &req);
        printf("Done sending data on 0.\n");
    } else if (rank == 1) {
        printf("About to recv data on 1\n");
        communicator->irecv(to_recv, 5*sizeof(int), 0, 50, &req);
        printf("Done recv data on 1\n");
    }
    
    if(rank == 0 || rank == 1) {
        
        printf("Calling wait on rank %d...\n", rank);
        
        MLCommon::cumlCommunicator::request_t *reqs = new MLCommon::cumlCommunicator::request_t[1] { req };
        communicator->waitall(1, reqs);
        
        printf("Done.\n");
        
        delete reqs;
    }
    
    if(rank == 1) {
        
        printf("RECEIVED: [");
        for(int i = 0; i < 5; i++) {
            
            printf("%d, ", to_recv[i]);
        }
        printf("]\n");
    }
    
    delete(to_send);
    free(to_recv);

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
    
    
void test_ep(void *ep) {

    std::cout << "Testing EP: " << ep << std::endl;
//     ucp_ep_print_info(*((ucp_ep_h*)ep), stdout);
}

void inject_comms_py(ML::cumlHandle *handle, ncclComm_t comm, void *ucp_worker, void *eps, int size, int rank) {
    
    ucp_worker_print_info((ucp_worker_h)ucp_worker, stdout);

    ucp_ep_h *new_ep_arr = new ucp_ep_h[size];
    
    size_t *size_t_ep_arr = (size_t*)eps;
    
    for(int i = 0; i < size; i++) {
        
        size_t ptr = size_t_ep_arr[i];
        if(ptr != 0) {
            ucp_ep_h *eps_ptr = (ucp_ep_h*)size_t_ep_arr[i];
            new_ep_arr[i] = *eps_ptr;
        } else {
            new_ep_arr[i] = nullptr;
        }
    }

    inject_comms(*handle, comm, (ucp_worker_h)ucp_worker, (ucp_ep_h*) new_ep_arr, size, rank);
}
    


} // end namespace HelloMPI
