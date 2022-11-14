from Cython.Build import cythonize

from setuptools import setup, Extension

extensions = [
    Extension("bintablefile", ["bintablefile/__init__.pyx"]),
]
extensions = cythonize(extensions)

long_description = open('README.md', "rt").read()
with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")
setup(
    ext_modules=extensions,
    name="bintablefile",
    version="2.0.0",
    packages=['bintablefile'],
    author="andrei.suiu/marian.rusu",
    description="Binary Table File - efficient binary file format to store and retrieve tabular data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/eSAMTrade/bintablefile",
    python_requires='>=3.7',
    setup_requires=["cython", "numpy>=1.13.0"],
    # package_data is required due to this bug:
    #   https://stackoverflow.com/questions/60717795/python-setup-py-sdist-with-cython-extension-pyx-doesnt-match-any-files
    package_data={"bintablefile": ["__init__.pyx"]},
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
)
