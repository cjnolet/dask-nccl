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
NcclClique *create_clique(ncclComm_t comm, int workerId, int nWorkers) {

    printf("Creating clique with comm=%s, nWorkers=%d, worker=%d\n", comm, nWorkers, workerId);

    int rank = -1;

    NCCL_CHECK(ncclCommUserRank(comm, &rank));

    printf("Verified rank = %d\n", rank);

    ML::cumlHandle *handle = new ML::cumlHandle(); // in this example, the NcclClique will take ownership of handle.
    initialize_comms(*handle, comm, nWorkers, workerId);

    return new NcclClique(handle, workerId, nWorkers);
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
