# gas-impl
# GAS-SN: generalized alpha-stable distribution with skew-normal family

## Overview - Fractional Distributions

This package is upgraded to incorporate the skew-normal family in 2025, called GAS-SN. 
The GAS-SN distribution is the most flexible distribution up to date, that can fit data sets with high skewness and kurtosis.

The content is written in a book format, located at [here](docs/fracdist.pdf).
As of July 2026, the book is in late editing stage.
However, I just discovered that the fractional gamma distribution is equivalent
to the inverse power of tilted stable law (Luc Devroye (2009)).
The book might go through another revision, most likely ceoncept simplification.

This package provides the reference implementation for all the distributions mentioned in the book.
See the test cases for usage examples. Every function is tested in at least one test cases.

In late 2025, I have decided to call this group of new distributions **The Fractional Distributions**.
Many classic distributions are generalized into the fractional distributions such as:
<table>
<tr>
<th> classic distribution </th><th> fractional distribution </th>
</tr><tr>
<td> generalized gamma distribution </td><td> fractional gamma distribution (Chapter 6)</td>
</tr><tr>
<td> chi/chi2 distribution </td><td> fractional chi/chi2 distribution (Chapter 7)</td>
</tr><tr>
<td> F distribution </td><td> fractional F distribution (Chapter 8)</td>
</tr><tr>
<td> univariate skew-t distribution </td><td> univariate GAS-SN distribution (Chapter 12)</td>
</tr><tr>
<td> stable / chauchy distributions  </td><td> ditto</td>
</tr><tr>
<td> multivariate skew-t distribution </td><td> multivariate elliptical GAS-SN distribution (Chapter 15)</td>
<tr>
</table>

## Physical Interpretation

This distribution family has the following parameters and they have physical meanings in the fractional transport problems, representing self-similar anomalous diffusion.

* $\alpha$ is the Levy index that acts as the temporal scaling exponent.
* $p$ (fractional gamma distribution) acts as the spatial scaling exponent

In the fractional chi/chi2 distribution, $p = \alpha$.
The space-scale and time-scale are synchronized. The PDF retains its exact shape as it dilates over time.
This makes the distribution structurally invariant under scaling transformations, 
establishing a bridge to fractional Brownian motion.

When $\alpha = 1$

* $k$ is the degrees of freedom. It encapsulates the spatial dimension or fractal topology of the network.



## Demo

To showcase how good the fits are, two univariate fits on VIX and SPX return distributions are presented below:
<table>
<tr>
<td> <img src="docs/plot_vix_gas_sn.png"> </td>
<td> <img src="docs/plot_spx_gas_sn.png"> </td>
<tr>
</table>

The bivariate VIX-SPX data set is fitted with the bivariate elliptical distribution, as shown below:

<img src="docs/plot_elliptical_vix_spx.png" width="600">

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
