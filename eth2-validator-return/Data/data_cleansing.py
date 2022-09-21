import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
import seaborn as sns

def plot_cdf(data, var, lb, ub, increment, title):
    mybins = np.append(-np.inf, np.arange(lb, ub, increment))
    mybins = np.append(mybins, np.inf)
    mybins = np.round(mybins, 2)
    count, bins = np.histogram(data[var], bins=mybins)
    table = pd.DataFrame({'bin': bins[1:], 'count': count})
    table["pdf"] = table["count"] / sum(table["count"])
    table["cdf"] = np.cumsum(table["pdf"])
    
    # plotting CDF
    plt.plot(table['bin'], table["cdf"])
    plt.xlabel(var)
    plt.ylabel('Cumulative percentage')
    plt.title(title)
    return table

def format_date(x):
    return datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')

def format_date_m(x):
    return datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M')

def format_date_ms(x):
    return datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S.%f')

def format_date_h(x):
    return datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H')

def format_date_d(x):
    return datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d')

####################
## Data Cleansing ##
####################
rev_data = pd.read_csv('Data/full_without_block_reward_fixed.csv')

## Sort data by block number
rev_data = rev_data.sort_values(by=['block_number'], ascending = True)

## check missing blocks
check = pd.DataFrame({'block_number':rev_data['block_number'], 'interval':rev_data['block_number'].shift(-1) - rev_data['block_number']})
missing_block = check[check['interval']!=1]
# aa = rev_data[rev_data['block_number'].isin([13620699, 13620700, 13620701, 13620702])]

## Check primary key - 14159890, 14526389 duplicated
np.unique(rev_data['block_number'].duplicated())    

## 0 timestamp & n.a.
missing_timestamp = rev_data[rev_data['block_timestamp']==0]
(rev_data['block_timestamp'].isna()).value_counts()

## Assign missing timestamp
rev_data.loc[rev_data['block_number']==13274226, 'block_timestamp'] = 1632295479
rev_data.loc[rev_data['block_number']==13842648, 'block_timestamp'] = 1640011369
rev_data.loc[rev_data['block_number']==13849751, 'block_timestamp'] = 1640105938
rev_data.loc[rev_data['block_number']==14680870, 'block_timestamp'] = 1651257329
rev_data.loc[rev_data['block_number']==14681716, 'block_timestamp'] = 1651269088
rev_data.loc[rev_data['block_number']==14686142, 'block_timestamp'] = 1651329205
rev_data.loc[rev_data['block_number']==14752821, 'block_timestamp'] = 1652244705
rev_data.loc[rev_data['block_number']==14762007, 'block_timestamp'] = 1652371328

## Correct wrong timestamp
rev_data.loc[rev_data['block_number']==13773001, 'block_timestamp'] = 1639079742
rev_data.loc[rev_data['block_number']==13773002, 'block_timestamp'] = 1639079781
rev_data.loc[rev_data['block_number']==13773003, 'block_timestamp'] = 1639079784
rev_data.loc[rev_data['block_number']==13773004, 'block_timestamp'] = 1639079806
rev_data.loc[rev_data['block_number']==13773005, 'block_timestamp'] = 1639079862

## Correct wrong block_net_profit
# rev_data.loc[rev_data['block_number']==13193116, 'block_net_profit'] = 0.341129
# rev_data.loc[rev_data['block_number']==12965107, 'block_net_profit'] = 0.023594
# rev_data.loc[rev_data['block_number']==13223730, 'block_net_profit'] = 0.132679
# rev_data.loc[rev_data['block_number']==14953916, 'block_net_profit'] = 11.52510
# rev_data.loc[rev_data['block_number']==13911172, 'block_net_profit'] = 0.239266
# rev_data.loc[rev_data['block_number']==13222409, 'block_net_profit'] = 0.553979
# rev_data.loc[rev_data['block_number']==14953917, 'block_net_profit'] = 6.261863
# rev_data.loc[rev_data['block_number']==14781918, 'block_net_profit'] = 1.100639
# rev_data.loc[rev_data['block_number']==13417951, 'block_net_profit'] = 15.215966
# rev_data.loc[rev_data['block_number']==14953915, 'block_net_profit'] = 0.102813
# rev_data.loc[rev_data['block_number']==14023027, 'block_net_profit'] = 0.551567
# rev_data.loc[rev_data['block_number']==14171298, 'block_net_profit'] = 0.117129
# rev_data.loc[rev_data['block_number']==14177901, 'block_net_profit'] = 0.073741


## Convert to eth
rev_data['tail_gas_price'] = rev_data['tail_gas_price']/1e18
rev_data['base_fee_per_gas'] = rev_data['base_fee_per_gas']/1e18
rev_data['burnt_fee'] = rev_data['base_fee_per_gas']*rev_data['total_gas_used']

## Convert timestamp to date format
rev_data['datetime'] = rev_data['block_timestamp'].apply(format_date)
rev_data['datetime_d'] = rev_data['block_timestamp'].apply(format_date_d)
rev_data['datetime_h'] = rev_data['block_timestamp'].apply(format_date_h)


## Extract year, month, day, hour, min, sec
rev_data['datetime'] = pd.to_datetime(rev_data['datetime'], format='%Y-%m-%d %H:%M:%S')
rev_data['year'] = rev_data.datetime.dt.year
rev_data['month'] = rev_data.datetime.dt.month
rev_data['week'] = rev_data.datetime.dt.isocalendar().week
rev_data['week_in_month'] = ((rev_data.datetime.dt.day - 1)/7).astype(int) + 1
rev_data['day'] = rev_data.datetime.dt.day
rev_data['hour'] = rev_data.datetime.dt.hour
rev_data['minute'] = rev_data.datetime.dt.minute
rev_data['second'] = rev_data.datetime.dt.second
    
## Create block length in seconds
rev_data['block_time'] = rev_data['block_timestamp'].shift(-1) - rev_data['block_timestamp']

## Remove last row n.a. block time
rev_data = rev_data[np.invert(rev_data['block_time'].isna())]

## Create profit for non-FB bundles
rev_data['profit_no_bundles'] = np.where(rev_data['block_net_profit']>0, rev_data['block_net_profit'] - rev_data['profit_from_bundles'], rev_data['block_net_profit'])

## Create FB-bundle dummy flag
rev_data['bundles_dummy'] = np.where(rev_data['profit_from_bundles']>0, 1, 0)

## Create FB dataset
fb_data = rev_data[rev_data['bundles_dummy']==1]

## Calculate profit per second
rev_data['profit_per_second'] = rev_data['block_net_profit'] / rev_data['block_time']

# mylist = [13274226, 13842648, 13849751, 14680870, 14681716, 14686142, 14752821, 14762007, 
#           13773001, 13773002, 13773003, 13773004, 13773005,
#           15177320, 15177478]
# check = rev_data[rev_data['block_number'].isin(mylist)]

# check_outlier = rev_data.sort_values(by=['block_net_profit'], ascending = False)
# check_outlier = check_outlier.head(20)


# freq_table = plot_cdf(data=rev_data, var='block_net_profit', lb=0, ub=5.1, increment=0.05, title='CDF of block net profit under 5 ETH')
rev_data.to_csv("clean_rev.csv")