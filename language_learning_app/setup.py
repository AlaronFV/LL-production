import sys
from setuptools import setup, Extension
import pybind11
import numpy

# Define the C++ extension module
ext_modules = [
    Extension(
        # The name of the extension module in Python
        'i_plus_one.i_plus_one_cpp',
        # List of C++ source files
        [
            'i_plus_one/bindings.cpp',
            'i_plus_one/vocab_model.cpp',
            'i_plus_one/llmodel.cpp',
            'i_plus_one/learning_queue.cpp',
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to numpy headers
            numpy.get_include(),
            # Path to our own C++ headers
            'i_plus_one/cpp_headers',
        ],
        language='c++',
        # Add compiler flags for C++17
        extra_compile_args=['-std=c++17'] if sys.platform != 'win32' else ['/std:c++17', '/permissive-'],
    ),
]

setup(
    name="i_plus_one",
    version="0.1.0",
    packages=["i_plus_one"],
    ext_modules=ext_modules,
    # Add pybind11 to setup_requires
    setup_requires=["pybind11>=2.10", "numpy>=1.20"],
    install_requires=[
        "numpy>=1.20"
    ],
    zip_safe=False,
)