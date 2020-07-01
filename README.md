# mahts

[![Build Status][travis-image]][travis-url]  [![PyPI version][pypi-image]][pypi-url]  [![PyPI download][download-image]][pypi-url]


Model Agnostic Hierarchical Time Series in Python

This package implements algorithms for Hierarchical Times Series reconciliation. It is based on the [hts R package](https://cran.r-project.org/web/packages/hts/index.html) ([Paper](https://cran.r-project.org/web/packages/hts/vignettes/hts.pdf)) but has the benefit of being **agnostic to the forecasting model**.

## Key Features
* Easy to use interface.
* Standard top-down, bottom-up & forecast-proportion approaches.
* Fast computation of optimal combination through sparse matrices.
* Optimal combination approach supports WLS with custom weights.
* Optimal combination approach supports bounded solutions through the Trust Region Reflective algorithm.
* Agnostic to the forecasting model.

## Installation

Via [PyPI](https://pypi.org/project/hausdorff/):

```bash
pip install mahts
```
Or you can clone this repository and install it from source: 

```bash
python setup.py install
```


[travis-image]: https://travis-ci.org/mavillan/mahts.svg?branch=master
[travis-url]: https://travis-ci.org/mavillan/mahts
[pypi-image]: http://img.shields.io/pypi/v/mahts.svg
[download-image]: http://img.shields.io/pypi/dm/mahts.svg
[pypi-url]: https://pypi.org/project/mahts


