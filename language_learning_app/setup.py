from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = cythonize([
    Extension("cy_utils.vocab_model", ["cy_utils/vocab_model.pyx"],
              include_dirs=[".", np.get_include()], language="c++", extra_compile_args=["-std=c++17"]),
    Extension("cy_utils.llmodel",    ["cy_utils/llmodel.pyx"],
              include_dirs=[".", np.get_include()], language="c++", extra_compile_args=["-std=c++17"]),
    Extension("cy_utils.queue",      ["cy_utils/queue.pyx"],
              include_dirs=[".", np.get_include()], language="c++", extra_compile_args=["-std=c++17"]),
], compiler_directives={"language_level": "3"})

setup(
    name="i_plus_one",
    version="0.1.0",
    packages=["cy_utils"],
    ext_modules=exts,
    install_requires=[
        "numpy>=1.20"
    ],
    setup_requires=["Cython","numpy>=1.20"],
    zip_safe=False,
)