from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

cuda_include_dir = '/usr/local/cuda/include'
cuda_lib_dir = "/usr/local/cuda/lib64"

srcs = ["nccl_example.pyx", "nccl_example_c.cpp"]
extensions = [Extension("nccl_example",
                        sources=srcs,
                        language="c++",
                        libraries=["cuda", "nccl"],
                        include_dirs=[cuda_include_dir, "/share/conda/cuml4/include"],
                        runtime_lib_dirs = [cuda_lib_dir],
                        extra_compile_args=["-std=c++11"])]
setup(name="nccl_example", ext_modules = cythonize(extensions))