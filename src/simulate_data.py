import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# We will set a constant random seed to make the result reproducible
seed = 1
np.random.seed(seed)


def simulate_fx_data(n_currencies=20):
    # For simplicity we'll consider that all currencies observe the same timezone - LDN timezone
    # We will simulate the hourly prices of currencies assuming they follow a geometric brownian motion
    # with different mean over different time buckets.
    # The prices are quoted as CCYUSD,
    # i.e positive returns indicate the currency is appreciating against USD.

    # As per the article, we have 4 different time buckets:
    # We simulate trading of currencies from 00:00 to 10PM
    # Local session will start from 7AM to 2PM - We will here simulate a negative drift
    # Overlap with US session will be from 2PM to 6PM - We will here simulate a negative drift
    # US session only will be from 6PM to 10PM - We will here simulate a positive drift
    # Non US session will be from 00AM to 7AM - We will here simulate a random walk with no drift

    dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='B')
    n_days = len(dates)
    # Generate daily volatilities for each currency, selected randomly between 0.2% and 1% per day
    # We then convert in hourly volatilities.
    volatilities = np.random.uniform(0.002, 0.01, n_currencies)
    hourly_volatilities = volatilities / np.sqrt(24)

    # Simulate the negative mean hourly returns
    mu_neg = -hourly_volatilities / 4
    neg_rets = simulate_gbm_returns(mu_neg, hourly_volatilities, n_days, hours=np.arange(7, 18), dates=dates)
    # Simulate the positive mean hourly returns
    # We adjust the mean by the respective size of positive period vs negative periods
    mu_pos = -mu_neg * 11 / 4
    pos_rets = simulate_gbm_returns(mu_pos, hourly_volatilities, n_days, hours=np.arange(18, 22), dates=dates)
    # Simulate the zero mean hourly returns from 00 to 07
    neutral_rets_sod = simulate_gbm_returns(0, hourly_volatilities, n_days, hours=np.arange(0, 7), dates=dates)
    # Simulate the zero mean hourly returns from 22 to 00
    neutral_rets_eod = simulate_gbm_returns(0, hourly_volatilities, n_days, hours=np.arange(22, 24), dates=dates)

    # Combine all returns and form the dataframe of prices
    rets = pd.concat([neg_rets, pos_rets, neutral_rets_sod, neutral_rets_eod], axis=0).sort_index()
    intraday_prices = np.exp(np.cumsum(rets))
    # Shift the prices forward as the return of Hi are intended to be rets from Hi to Hi+1
    intraday_prices = intraday_prices.shift()
    intraday_prices = intraday_prices.loc[intraday_prices.index.hour <= 22]
    intraday_prices.iloc[0, :] = 1
    intraday_prices.columns = [f'ccy_{i}' for i in range(intraday_prices.shape[1])]
    return intraday_prices


def simulate_gbm_returns(mu, sigma, size, hours, dates):
    n_hours = len(hours)
    expanded_index = pd.to_datetime([f"{date} {hour}:00" for date in dates for hour in hours])
    rets = np.random.normal(loc=mu - (sigma ** 2) / 2, scale=sigma, size=(size * n_hours, len(sigma)))
    rets = pd.DataFrame(rets, index=expanded_index)
    return rets


def plot_price(data, n_plots=6):
    # We'll  visualize our simulated price to check that they are sensible
    data.iloc[:, :n_plots].plot()
    plt.title('Simulated Currency Price over time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
