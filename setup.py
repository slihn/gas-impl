from setuptools import setup

from gas_impl import __version__

setup(
    name='gas_impl',
    version=__version__,

    url='https://github.com/slihn/gas-impl',
    author='Stephen H. Lihn',
    author_email='stevelihn@gmail.com',

    py_modules=['gas_impl'],
    
    install_requires=[
        'numpy',
        'pandas',
        'pandas-stubs',
        'scipy',
        'scipy-stubs',
        'scikit-image',
        'pyro-ppl',
        'matplotlib',
        'mpmath',
        'tsquad',
        'numba',
        'pandarallel',
        'diskcache',
        'loguru',
    ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'License :: Free for non-commercial use',
    ],
)
