__VERSION__ = "2.2.0"

from distutils.command.build import build as build_orig

from setuptools import setup, Extension


class build(build_orig):
    """
    We require this due to the fact that when setup runs, cython is not installed yet
    See: https://stackoverflow.com/questions/60717795/python-setup-py-sdist-with-cython-extension-pyx-doesnt-match-any-files
    """

    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        from Cython.Build import cythonize
        # By default, cythonize uses Py2.x for *.pyx files, so we need to force it to use Py3.x
        LANGUAGE_LEVEL = 3
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules, language_level=LANGUAGE_LEVEL, annotate=True)


exts = [Extension(name="bintablefile",
                  sources=["bintablefile/bintablefile.pyx"],
                  include_dirs=[])]

long_description = open('README.md', "rt").read()
with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")

setup(
    ext_modules=exts,
    name="bintablefile",
    version=__VERSION__,
    packages=['bintablefile'],
    author="andrei.suiu/marian.rusu",
    description="Binary Table File - efficient binary file format to store and retrieve tabular data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/eSAMTrade/bintablefile",
    python_requires='>=3.7',
    setup_requires=["cython", "numpy>=1.13.0"],
    # package_data is required due to issues with setuptools and cython
    #   https://stackoverflow.com/questions/60717795/python-setup-py-sdist-with-cython-extension-pyx-doesnt-match-any-files
    package_data={"bintablefile": ["bintablefile.pyx"], },
    # If we don't include these, these files won't get into the sdist package
    data_files=[(".", ["requirements.txt", "requirements-dev.txt"])],
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
    cmdclass={"build": build},
)
