#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <stdexcept>

#define THROW(fmt, ...)                                         \
    do {                                                        \
        std::string msg;                                        \
        char errMsg[2048];                                      \
        sprintf(errMsg, "Exception occured! file=%s line=%d: ", \
                __FILE__, __LINE__);                            \
        msg += errMsg;                                          \
        sprintf(errMsg, fmt, ##__VA_ARGS__);                    \
        msg += errMsg;                                          \
        throw std::runtime_error(msg);                          \
    } while(0)

#define ASSERT(check, fmt, ...)                  \
    do {                                         \
        if(!(check))  THROW(fmt, ##__VA_ARGS__); \
    } while(0)

#define COMM_CHECK(call)                                        \
  do {                                                          \
    auto status = call;                                         \
    ASSERT(status == 0, "FAIL: call='%s'!", #call);             \
  } while (0)





#pragma once


namespace HelloMPI {

class MpiWorldBuilder {
public:
    /**
     * @brief ctor
     * @param wid worker Id as received from dask world. It thus expects that
     * the dask-workers have been created with some way of identifying their IDs!
     * @param num number of workers
     */
    MpiWorldBuilder(int wid, int num):
        workerId(wid), nWorkers(num), portName() {
        workerId == 0? openPort() : getPort();
        workerId == 0? connectToClients() : connectToServer();
    }

    /** dtor */
    ~MpiWorldBuilder() {
        if(workerId == 0) {
            printf("worker=%d: server closing port\n", workerId);
            COMM_CHECK(MPI_Close_port(portName.c_str()));
        }
    }
    
        /** comm handle for all the connected processes so far */
    MPI_Comm intracomm;


private:
    /** current dask worker ID received from python world */
    int workerId;
    /** number of workers */
    int nWorkers;
    /** port name returned by MPI_Open_port */
    std::string portName;
    /** comm handle for the current newly connected MPI_Comm */
    MPI_Comm intercomm;

    void openPort() {
        char _portName[MPI_MAX_PORT_NAME];
        COMM_CHECK(MPI_Open_port(MPI_INFO_NULL, _portName));
        portName = _portName;
        printf("worker=%d port opened on %s\n", workerId, portName.c_str());
        COMM_CHECK(MPI_Publish_name("server", MPI_INFO_NULL, _portName));
    }

    void getPort() {
        sleep(1); // wait for the server to publish!
        char _portName[MPI_MAX_PORT_NAME];
        COMM_CHECK(MPI_Lookup_name("server", MPI_INFO_NULL, _portName));
        portName = _portName;
        printf("worker=%d server port obtained to be %s\n", workerId,
               portName.c_str());
    }

    void connectToClients() {
        for(int i=1;i<nWorkers;++i) {
            printf("worker=%d: server: trying to connect to client=%d\n",
                   workerId, i);
            COMM_CHECK(MPI_Comm_accept(portName.c_str(), MPI_INFO_NULL, 0,
                                       i == 1? MPI_COMM_WORLD : intracomm,
                                       &intercomm));
            printf("worker=%d: server: accepted connection from client=%d\n",
                   workerId, i);
            printf("worker=%d: server: merging intercomm\n", workerId);
            COMM_CHECK(MPI_Intercomm_merge(intercomm, 0, &intracomm));
            printf("worker=%d: server: intercomm merged\n", workerId);
            int rank, nranks;
            MPI_Comm_rank(intracomm, &rank);
            MPI_Comm_size(intracomm, &nranks);
            printf("worker=%d: server: after merging from client=%d rank=%d/%d\n",
                   workerId, i, rank, nranks);
        }
    }

    void connectToServer() {
        printf("worker=%d: client: trying to connect to server %s\n",
               workerId, portName.c_str());
        
        COMM_CHECK(MPI_Comm_connect(portName.c_str(), MPI_INFO_NULL, 0,
                                    MPI_COMM_WORLD, &intercomm));
        
        printf("worker=%d: client: connected to server\n", workerId);
        printf("worker=%d: client: merging intercomm\n", workerId);
        
        
        COMM_CHECK(MPI_Intercomm_merge(intercomm, 1, &intracomm));
        
        printf("worker=%d: client: intercomm is now merged\n", workerId);
        int rank, nranks;
        MPI_Comm_rank(intracomm, &rank);
        MPI_Comm_size(intracomm, &nranks);
        printf("worker=%d: client: after merging with server rank=%d/%d\n",
               workerId, rank, nranks);
        // merge the other workers as well in the clients
        for(int i=nranks;i<nWorkers;++i) {
            printf("worker=%d: client: trying to accept for client=%d\n",
                   workerId, i);
            COMM_CHECK(MPI_Comm_accept(portName.c_str(), MPI_INFO_NULL, 0,
                                       intracomm, &intercomm));
            printf("worker=%d: client: accepted connection from client=%d\n",
                   workerId, i);
            printf("worker=%d: client: merging intercomm\n", workerId);
            COMM_CHECK(MPI_Intercomm_merge(intercomm, 0, &intracomm));
            printf("worker=%d: client: intercomm is now merged\n", workerId);
            int rank, nranks;
            MPI_Comm_rank(intracomm, &rank);
            MPI_Comm_size(intracomm, &nranks);
            printf("worker=%d: client: after merging from client=%d rank=%d/%d\n",
                   workerId, i, rank, nranks);
        }
    }
};
    
MpiWorldBuilder *mpi_run(int workerId, int nWorkers);
void mpi_init(int workerId);
void mpi_finalize(int workerId);
int get_rank(MpiWorldBuilder *comm,int workerId);

}; // end namespace HelloMPI
