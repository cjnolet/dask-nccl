import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

cuda_include_dir = '/usr/local/cuda/include'
cuda_lib_dir = "/usr/local/cuda/lib64"
if os.environ.get('CUDA_HOME', False):
    cuda_lib_dir = os.path.join(os.environ.get('CUDA_HOME'), 'lib64')
    cuda_include_dir = os.path.join(os.environ.get('CUDA_HOME'), 'include')

include_dirs = [cuda_include_dir]
if os.environ.get("CONDA_PREFIX", None):
    include_dirs.append(os.environ.get("CONDA_PREFIX") + "/include")

srcs = ["nccl-example/nccl_example.pyx", "nccl-example/nccl_example_c.cpp"]

extensions = [Extension("nccl_example",
                        sources=srcs,
                        language="c++",
                        libraries=["cuda", "nccl"],
                        include_dirs=include_dirs,
                        runtime_lib_dirs = [cuda_lib_dir],
                        extra_compile_args=["-std=c++11"])]

setup(name="nccl_example", ext_modules = cythonize(extensions))