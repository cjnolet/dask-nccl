import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib

cuda_include_dir = '/usr/local/cuda/include'
cuda_lib_dir = "/usr/local/cuda/lib64"
if os.environ.get('CUDA_HOME', False):
    cuda_lib_dir = os.path.join(os.environ.get('CUDA_HOME'), 'lib64')
    cuda_include_dir = os.path.join(os.environ.get('CUDA_HOME'), 'include')

print(os.environ.get("CUML_HOME"))

if os.environ.get("CUML_HOME", False):
    prims_include = os.path.join(os.environ.get("CUML_HOME"), "cpp/src_prims")
    cuml_include = os.path.join(os.environ.get("CUML_HOME"), "cpp/src")
    comms_include = os.path.join(os.environ.get("CUML_HOME"), "cpp/nccl_comms/include")

    cuml_lib = os.path.join(os.environ.get("CUML_HOME"), "cpp/build")
    comms_lib = os.path.join(os.environ.get("CUML_HOME"), "cpp/nccl_comms/build")

    print(cuml_lib)
else:
    print("Need to set CUML_HOME environment variable to location of cuML source")

include_dirs = [cuda_include_dir, prims_include, cuml_include, comms_include]
library_dirs = [cuda_lib_dir, cuml_lib, comms_lib]

if os.environ.get("CONDA_PREFIX", None):
    include_dirs.append(os.environ.get("CONDA_PREFIX") + "/include")

srcs = ["nccl-example/nccl_example.pyx", "nccl-example/nccl_example_c.cpp"]

extensions = [Extension("nccl_example",
                        sources=srcs,
                        language="c++",
                        libraries=["cuda", "nccl", "cuml++", "cumlcomms"],
                        include_dirs=include_dirs,
                        library_dirs=library_dirs,
                        extra_compile_args=["-std=c++11"]),
              ]
setup(name="nccl_example", ext_modules = cythonize(extensions))