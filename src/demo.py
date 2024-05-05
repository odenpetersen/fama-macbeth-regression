#!/usr/bin/env python3
import glob
import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt

def get_df():
    return pd.read_csv('../data/NIFTY50_all.csv')

def get_panels(df=None,variable='VWAP'):
    if df is None:
        df = get_df()
    df = df[('2018-00-00' < df.Date) & (df.Date < '2020-00-00')]
    df = df[['Date','Symbol',variable]].set_index('Date').pivot(columns='Symbol')
    df.columns = df.columns.get_level_values(1)
    return df

def get_returns(panels=None):
    if panels is None:
        panels = get_panels()
    return panels.pct_change().iloc[1:]

def get_eigenportfolios(returns=None):
    if returns is None:
        returns = get_returns()
    eigenportfolios, _, _ = np.linalg.svd((returns-returns.mean()), full_matrices=False)
    eigenportfolios = pd.DataFrame(eigenportfolios, index = returns.index)
    return eigenportfolios

#Step one of FM regression. Compute the exposure of each asset to the risk factors.
def get_betas(returns, risk_factors):
    assert len(set([tuple(df.columns) for df in risk_factors.values()]))==1
    exposures = dict()
    for asset in returns:
        model = sklearn.linear_model.LinearRegression()
        model.fit(risk_factors[asset], returns[asset])
        exposures[asset] = [*model.coef_, model.intercept_]
    return pd.DataFrame(exposures, index=[*risk_factors[asset].columns, 'constant_factor'])

#Step two of FM regression. Compute the risk premia at each point in time.
def get_premia(returns, betas):
    premia = dict()
    for date in returns.index:
        model = sklearn.linear_model.LinearRegression()
        model.fit(betas.T, returns.loc[date])
        premia[date] = [*model.coef_, model.intercept_]
    return pd.DataFrame(premia, index=[*betas.index, 'equal_weighting'])

def fama_macbeth(returns, risk_factors):
    betas = get_betas(returns, risk_factors)
    premia = get_premia(returns, betas)
    return betas, premia

def demo_eigenportfolios():
    returns = get_returns()
    risk_factors = get_eigenportfolios(returns)
    risk_factors = {asset : risk_factors for asset in returns}
    betas, premia = fama_macbeth(returns, risk_factors)
    premia_subset = premia.T[[*premia.T.columns[:3],*premia.T.columns[-5:]]]

    (np.sign(premia_subset.mean())*premia_subset/premia_subset.std()).cumsum().plot()
    plt.suptitle('Cumulative Excess Return of Eigenportfolios')
    plt.title('No compounding, normalised to variance=1')
    plt.savefig('../output/eigenportfolios.png')

def demo_features():
    df = get_df()
    panels = {variable : get_panels(df=df,variable=variable) for variable in df.columns[3:]}
    returns = get_returns(panels['VWAP'])

    def get_features(asset, panels=panels):
        panels = {variable : panels[variable][asset] for variable in panels}
        #Have tried to make sure that these don't have lookahead. Assumption is that we'll put on a position throughout the day at VWAP and then offload it the next day, so we can only use information available pre-open.
        features = dict()

        features['overnight'] = (panels['Open'] - panels['Prev Close'])
        features['overnight_squared'] = features['overnight']**2
        features['intraday'] = (panels['Close'] - panels['Open'])
        features['intraday_squared'] = features['intraday']**2
        features['high_minus_low'] = panels['High']-panels['Low']
        features['close_minus_last'] = panels['Close']-panels['Last']
        features['volume'] = panels['Volume']
        features['turnover'] = panels['Turnover']
        features['trades'] = panels['Trades']
        features['deliverable_volume'] = panels['Deliverable Volume']
        features['deliverable_volume_pct'] = panels['%Deliverble']

        features = pd.DataFrame(features)
        features = features.shift(1)
        features = features.loc[returns.index]
        return features

    risk_factors = {asset : get_features(asset) for asset in returns}
    market_factor = (lambda x : x*np.sign(x.mean()))(get_eigenportfolios(returns)[0]) #This is heuristic naming
    for asset in risk_factors:
        risk_factors[asset]['market_factor'] = market_factor

    betas, premia = fama_macbeth(returns, risk_factors)

    (np.sign(premia.T.mean())*premia.T/premia.T.std()).cumsum().plot()
    plt.suptitle('Cumulative Excess Return of Feature-Based Risk Factors')
    plt.title('No compounding, normalised to variance=1')
    plt.savefig('../output/features.png')

if __name__ == '__main__':
    demo_eigenportfolios()
    demo_features()
