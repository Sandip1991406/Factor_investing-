# LIBRARIES
import pickle
import pandas as pd
import numpy as np
import pyfolio as pf
from multi_factor_util import get_performance_metrics, get_data_from_dict
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

# DATA LOADING
data_filename = 'multifactor_data_2017_2022.bz2'
with open(data_filename, 'rb') as file:
    data = pickle.load(file)

# EXTRACTING DATA
close_prices = get_data_from_dict(data, "Close")
total_equity = get_data_from_dict(data, "Total Equity")
total_liabilities = get_data_from_dict(data, "Total Liabilities")

close_prices.to_csv("close_prices.csv", index=False)
total_equity.to_csv("total_equity.csv", index=False)
total_liabilities.to_csv("total_liabilities.csv", index=False)

# DEBT TO EQUITY RATIO
de_ratio = total_liabilities / total_equity

# RETURN ON EQUITY (ROE)
net_income = get_data_from_dict(data, "Net Income")
roe = net_income / total_equity

# PORTFOLIO CONSTRUCTION & REBALANCING
rebalancing_schedule = pd.DataFrame(index=roe.index)
rebalancing_schedule['is_start_of_month'] = rebalancing_schedule.index.to_series().dt.month != rebalancing_schedule.index.to_series().shift(1).dt.month
start_of_month_roe = roe[roe.index.isin(rebalancing_schedule[rebalancing_schedule['is_start_of_month']].index)]

# RANKING STOCKS BASED ON ROE
roe_ranks = start_of_month_roe.rank(ascending=False, axis=1)
roe_ranks.head()

# RANKING STOCKS BASED ON DE RATIO
start_of_month_de = de_ratio[de_ratio.index.isin(rebalancing_schedule[rebalancing_schedule['is_start_of_month']].index)]
de_ratio_ranks = start_of_month_de.rank(ascending=True, axis=1)
de_ratio_ranks.head()

# CALCULATING AND RANKING GROWTH RATE
start_of_month_net_income = net_income[net_income.index.isin(rebalancing_schedule[rebalancing_schedule['is_start_of_month']].index)]
net_income_growth = start_of_month_net_income.pct_change(3)
growth_rate_ranks = net_income_growth.rank(ascending=False, axis=1)
growth_rate_ranks.head()

# IMPLEMENTING STRATEGY
roe_ranks = roe_ranks.drop([roe_ranks.index[0], roe_ranks.index[1], roe_ranks.index[2]])
de_ratio_ranks = de_ratio_ranks.drop([de_ratio_ranks.index[0], de_ratio_ranks.index[1], de_ratio_ranks.index[2]])
growth_rate_ranks = growth_rate_ranks.dropna()

total_ranks = roe_ranks + de_ratio_ranks + growth_rate_ranks
total_ranks.head()

total_ranks = total_ranks.rank(ascending=True, axis=1)
top_ranks = total_ranks[total_ranks <= 10]
top_ranks.head()

# SIGNAL GENERATION
rebalancing_schedule = pd.DataFrame(index=close_prices.index)
rebalancing_schedule['is_start_of_month'] = rebalancing_schedule.index.to_series().dt.month != rebalancing_schedule.index.to_series().shift(1).dt.month

monthly_signals = top_ranks.applymap(lambda x: 1 if x <= 10 else x)
monthly_signals.head()

monthly_signals.fillna(0, inplace=True)
daily_signals = monthly_signals.reindex(rebalancing_schedule.index)
daily_signals = daily_signals.ffill()
daily_signals.tail()

# CALCULATING DAILY RETURNS
daily_returns = close_prices.pct_change(axis=0)
strategy_returns = daily_signals.shift(1) * daily_returns
strategy_returns.dropna(inplace=True)
strategy_returns['Mean_Returns'] = strategy_returns.apply(lambda row: row[row != 0].mean(), axis=1)
strategy_returns.head()
strategy_returns.tail()

# PERFORMANCE ANALYSIS using PYFOLIO
get_performance_metrics(strategy_returns)

""" OUTPUT RESULT
                 Strategy
CAGR               14.31%
Sharpe Ratio          0.7
Maximum Drawdown  -25.07%  """




