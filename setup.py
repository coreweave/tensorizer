from setuptools import setup
from Cython.Build import cythonize

setup(
    include_dirs=["/usr/local/cuda/include"],
    ext_modules=cythonize("tensorizer/*.pyx"),
)
