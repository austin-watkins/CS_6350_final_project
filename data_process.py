import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime

if __name__ == '__main__':
    data_dir = 'data/'
    rpc_file = 'implanter_rtc.csv'
    apc_file = 'implanter_apc.csv'

    rpc = pd.read_csv(data_dir + rpc_file)
    apc = pd.read_csv(data_dir + apc_file)

    # remove any columns that only have nans
    rpc.dropna(axis=1, how='all', inplace=True)
    apc.dropna(axis=1, how='all', inplace=True)

    # Turn time columns into datetime.
    # There might be more data cleaning we want to do.
    rpc['time_stamp'] = pd.to_datetime(rpc['time_stamp'])
    apc['runstart'] = pd.to_datetime(apc['runstart'])

    # Sort by timestamp.
    rpc.sort_values('time_stamp', inplace=True)
    apc.sort_values('runstart', inplace=True)

    # todo convert main status codes to integers
    # ...

    # do some basic stats on the main status codes

    # make histogram on number of occurrences of status
    ax = rpc['main_status'].value_counts().plot.bar()
    ax.set_xlabel('status code')
    ax.set_ylabel('occurrences')
    plt.tight_layout()
    plt.savefig('status_code_bar_chart.png')
    plt.clf()

    # make histogram of amount of time spent in each status
    # todo make sure that duration is working the way I think it is (that it is for the correct status)
    status_values = rpc['main_status'].unique()
    status_to_time_spent = []
    for status_val in status_values:
        time_spent_in_status = rpc[rpc['main_status'] == status_val]['duration'].sum()
        # decimal in number of days
        time_spent_in_status *= 24
        status_to_time_spent.append((status_val, time_spent_in_status))
    status_to_time_spent = sorted(status_to_time_spent, key=lambda x: x[1], reverse=True)
    status_to_time_spent = pd.DataFrame(status_to_time_spent, columns=['main_status', 'duration'])
    status_to_time_spent.set_index('main_status', inplace=True)
    ax = status_to_time_spent.plot.bar()
    ax.set_xlabel('Status')
    ax.set_ylabel('Duration')
    plt.tight_layout()
    plt.savefig('time_spent_in_status.png')
    plt.clf()

