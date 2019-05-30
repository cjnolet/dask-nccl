# Dask + NCCL + cuML Comms PoC

This project is a proof of concept for running ad-hoc NCCL cliques within an existing Dask cluster. 

It provides a full end-to-end example, which
1. accepts Dask cuDF as input
2. calls down to a mock C++ "cuML algorithm" through cython 
3. performs a collective reduce across ranks
4. returns a Dask cuDF with the results of the reduction.

The included notebook (`demo.ipynb`) walks through a simple example of creating a class that wraps the maintenance logic for a `NcclClique` and performs the end-to-end demonstration at the end. 

This demo uses `LocalCUDACluster` to establish a cluster of workers with OPG access to the GPUs on a single node. Rather than using the `LocalCUDACluster`, the Dask client object can connect to an existing Dask cluster, such as one with workers that span physical nodes. This will be tested in next steps.

Dask is used to broadcast the NCCL `uniqueId` to the workers so that this proof of concept does not require a dependency on MPI. Next steps will also include demonstrating the use of UCX initialization using Dask workers. 

Steps to running this demonstration:

1. You will need to have NCCL2 and the Cuda Toolkit installed and available on your library and include paths. You 
can install nccl2 in conda with: 

`conda install -c nvidia nccl`

2. You can install cudatoolkit in your conda environment with:

`conda install cudatoolkit==10.0.130`

3. Check out the branch from the [cuML comms](https://github.com/rapidsai/cuml/pull/643) pull request and build the C++ source, as outlined in `BUILD.md` inside the cuml codebase. 

4. The cuML NCCL Communicator will need to be built and installed. You can find it in the pull request from step #3. The build instructions are outlined in the comments of the cuML comms PR.

5. Set the `CUML_HOME` environment variable to the location of the cuML source code.

6. To build the C++ and Cython portion of the demonstration, run the following in the project root directory:

`python setup.py install`


Run the demonstration notebook by executing the `jupyter notebook` or `jupyter lab` commands in the project root directory and navigate to the `demo.ipynb` notebook.

