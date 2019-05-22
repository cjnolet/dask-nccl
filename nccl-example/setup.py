from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os

if os.environ.get('CUDA_HOME', False):
    cuda_lib_dir = os.path.join(os.environ.get('CUDA_HOME'), 'lib64')
    cuda_include_dir = os.path.join(os.environ.get('CUDA_HOME'), 'include')

srcs = ["nccl_example.pyx", "nccl_example_c.cpp"]
extensions = [Extension("nccl_example",
                        sources=srcs,
                        language="c++",
                        libraries=["nccl", "cuda"],
                        runtime_library_dirs=[cuda_lib_dir],
                        include_dirs=[cuda_include_dir],
                        extra_compile_args=["-std=c++11"])]
setup(name="nccl_example",
      ext_modules = cythonize(extensions))
