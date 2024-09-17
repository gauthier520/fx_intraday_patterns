import matplotlib.pyplot as plt
import numpy as np


def plot_cum_mean_returns_per_hour(returns):
    # This function will plot the mean cumulative returns per hr of the day
    # in a grid for each currency, so we can visualize any pattern.
    mean_returns = returns.groupby(returns.index.hour).mean().cumsum()

    # Define the number of rows for the grid layout
    n_ccy = returns.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_ccy / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    axes = axes.flatten()
    # Plot for each ccy one by one
    for i in range(n_ccy):
        ccy_name = returns.columns[i]
        axes[i].plot(mean_returns.index, mean_returns.iloc[:, i] * 100, label=ccy_name)
        axes[i].set_title(f'Cumulative Mean Returns % - {ccy_name}')
        axes[i].set_xlabel('Hour')
        axes[i].set_ylabel('Ret (%)')
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].grid(True)

    plt.tight_layout()
    plt.legend()
    plt.show()


def sr_per_hour(returns):
    # Sharpe ratio of returns per hour
    # This will allow comparing results in a table with values having similar significance in order of magnitude
    sr_per_hr = returns.groupby(returns.index.hour).mean() / returns.groupby(returns.index.hour).std()
    sr_per_hr_global = sr_per_hr.mean(axis=1)
    sr_per_hr = sr_per_hr_global.rename('Mean across ccys').to_frame().join(sr_per_hr)
    return sr_per_hr
