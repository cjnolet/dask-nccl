#include "nccl_example_c.h"

#include <cuML.hpp>
#include <cuML_comms.hpp>

#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>

#include <stdio.h>
#include <unistd.h>
#include <string>
#include <stdexcept>

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

    ncclUniqueId id;
    memcpy(id.internal, uniqueId, NCCL_UNIQUE_ID_BYTES);

    /**
     * The following is meant to mimic the hand-off to an algorithm.
     *
     * Ideally, the cumlHandle would be passed into some helper function
     * to create and initialize the NCCL comms and the cumlHandle, itself,
     * passed into the follow-on algorithm to perform the MNMG work.
     */
    ML::cumlHandle *handle = new ML::cumlHandle();
    ncclComm_t comm;
    initialize_comms(*handle, comm, nWorkers, workerId, id);

    /**
     * A NcclClique mimicks the algorithm, which would use the cumlCommunicator
     * from the cumlHandle and perform the necessary collective comms.
     */
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



} // end namespace HelloMPI
