# gas-impl
# GAS-SN: generalized alpha-stable distribution with skew-normal family

## Overview

This package is upgraded to incorporate the skew-normal family in 2025, called GAS-SN. 
The GAS-SN distribution is the most flexible distribution up to date, that can fit data sets with high skewness and kurtosis.

The content is written in a book format, located at [here](docs/fracdist.pdf).
As of April 2025, the book is in early draft format. More detail will be filled in.

This package provides the reference implementation for all the distributions mentioned in the book.
See the test cases for usage examples. Every function is tested in at least one test cases.

To showcase how good the fits are, two univariate fits on VIX and SPX return distributions are presented below:
<table>
<tr>
<td> <img src="docs/plot_vix_gas_sn.png"> </td>
<td> <img src="docs/plot_spx_gas_sn.png"> </td>
<tr>
</table>

The bivariate VIX-SPX data set is fitted with the bivariate elliptical distribution, as shown below:

![VIX-SPX Elliptical Fit](docs/plot_elliptical_vix_spx.png)


My previous 2024 paper that laid the foundation is located at

https://arxiv.org/abs/2405.04693


## Installation

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

You can run the test cases to make sure everything is working in your platform:

```bash
cd /path-to/gas-impl
pytest
```

This package is developed under linux/ubuntu.
The 'pandarallel' package is used for multicore processing when an array is sent to the 'pdf' or 'cdf' functions.
This part of implementation may not work for a non-linux platform. Please be aware.

## Example

Once you have installed the package, you can run the following snippets:

```python
from gas_impl.stats import gsas

g = gsas(alpha=1.1, k=2.5)
g.pdf(0.25)
```

The answer should be 0.32678...

Add skewness with the beta parameter:

```python
from gas_impl.stats import gas_sn

g = gas_sn(alpha=1.1, k=2.5, beta=0.5)
g.pdf(0.25)
```

The answer should be 0.35881...
