#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include "hello_mpi_c.h"
#include <string>
#include <stdexcept>

namespace HelloMPI {

void mpi_init(int workerId) {
    
    printf("Init Invoked on %d!\n", workerId);
    
    int flag;
    MPI_Initialized(&flag);
    
    if(flag == 0)
        MPI_Init(NULL, NULL);
    else
        printf("MPI Already Initialized!");
}

    
MpiWorldBuilder *create_builder(int workerId, int nWorkers) {
    int rank, nranks;
    
    int flag;
    MPI_Initialized( &flag );
        
    printf("Is MPI Initialized? %d\n", flag);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    printf("Hello from rank=%d/%d worker=%d/%d\n", rank, nranks, workerId, nWorkers);

    MpiWorldBuilder *builder = new MpiWorldBuilder(workerId, nWorkers);
    return builder;
}
    
int get_rank(MpiWorldBuilder *comm) {
    int rank, nranks;
    MPI_Comm_rank(comm->intracomm, &rank);
    MPI_Comm_size(comm->intracomm, &nranks);
    printf("RANK: %d\n", rank);
    return rank;
}
    
void mpi_finalize(int workerId) {
    MPI_Finalize();
    printf("Finalize Invoked on %d!\n", workerId);
}

} // end namespace HelloMPI
