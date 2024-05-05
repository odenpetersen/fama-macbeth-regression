# Fama-MacBeth Regression
Python implementation of the [Two-Stage Fama-MacBeth Regression procedure](https://en.wikipedia.org/wiki/Fama%E2%80%93MacBeth_regression).

Two different choices of risk factors are demonstrated:

1. Eigenportfolios

![Eigenportfolios](https://github.com/odenpetersen/fama-macbeth-regression/blob/main/output/eigenportfolios.png?raw=true)

2. Microstructural features from the previous day, along with the first eigenportfolio

![Microstructural features](https://github.com/odenpetersen/fama-macbeth-regression/blob/main/output/features.png?raw=true)

# Data Source
Dataset is NIFTY50 from [Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) (version 15, as at 2024-05-05).

Note that this dataset has survivorship bias! Because of this, I only examine data from the beginning of 2018. I think it is less interesting to include at the effects of the 2020 correction for this simple analysis so I'll also exclude anything from the beginning of 2020 onwards.
