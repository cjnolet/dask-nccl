# Dask + NCCL PoC

This project is a proof of concept for running ad-hoc NCCL cliques within an existing Dask cluster.

The included notebook (`demo.ipynb`) performs the end-to-end demonstration. This demo uses `LocalCUDACluster` to 
establish a cluster of workers with OPG access to the GPUs on a single node and runs an allReduce for end-to-end verification.
Rather than using the `LocalCUDACluster`, the Dask client object can connect to an existing Dask cluster, such as one
with workers that span physical nodes. This will be tested in next steps. 

Dask is used to broadcast the NCCL `uniqueId` to the workers so that this proof of concept does not require a dependency on MPI. 

To run this demonstration, you will need to have NCCL2 and the Cuda Toolkit installed and available on your library and include paths. You 
can install nccl2 in conda with: 

`conda install -c nvidia nccl`

You can install cudatoolkit in your conda environment with:

`conda install cudatoolkit==10.0.130`

To build the C++ and Cython portion of the demonstration, run the following in the project root directory:

`python setup.py install`


Run the demonstration notebook by executing the `jupyter notebook` or `jupyter lab` commands in the project root directory.

