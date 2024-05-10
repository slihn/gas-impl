# gas-impl
# Reference implementation of generalized alpha-stable distribution

## Overview

This package provides the initial reference implementation
and test cases for my paper:

https://arxiv.org/abs/2405.04693

See the test cases for usage examples.

## Install

There is no plan to set up pypi yet. 

Please use the github local installation method:

```bash
pip install git+https://github.com/slihn/gas-impl.git#egg=gas_impl
```

If the above method doesn't work, a safer method is to 'git clone' this repository in a local directory. 
Then run:

```bash
cd /path-to/gas-impl
pip install -e .
```


## Example

Once you have installed the package, you can run the following snippet:

```python
from gas_impl.gas_dist import gsas

g = gsas(alpha=1.1, k=2.5)
g.pdf(0.25)
```

The answer should be 0.32678...


