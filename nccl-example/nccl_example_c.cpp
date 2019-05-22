#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include "nccl_example_c.h"
#include <string>
#include <stdexcept>

namespace NCCLExample {


NcclWorldBuilder *create_builder(int workerId, int nWorkers, char *uniqueId) {

    ncclUniqueId id;
    strncpy(id.internal, uniqueId, NCCL_UNIQUE_ID_BYTES);

    NcclWorldBuilder *builder = new NcclWorldBuilder(workerId, nWorkers, id);
    return builder;
}

char* get_unique_id() {

  ncclUniqueId id;
  ncclGetUniqueId(&id);

  char* newchar = id.internal;

  printf("unique_id=%s\n", newchar);

  return newchar;
}

} // end namespace HelloMPI
