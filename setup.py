import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib

extensions = [Extension("nccl_example",
                        sources= ["nccl-example/simple_reduce.pyx", "nccl-example/ucp_helper.cpp"],
                        language="c++",
                        libraries=["ucp", "ucs", "uct", "ucm"],
                        extra_compile_args=["-std=c++11"]),
              ]
setup(name="nccl_example", ext_modules = cythonize(extensions))
