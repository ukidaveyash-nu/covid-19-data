from header import *


def get_date(df, st_name):
    ret_df = pd.to_datetime(df[df['state'] == st_name]['date'])
    return ret_df.dt.strftime('%b/%d')


def get_cases(df, st_name):
    ret_df = df[df['state'] == st_name]['cases']
    return ret_df

def incr_change(df, format):
    temp = df.to_list()
    incr = []
    for i in range(len(temp)):
        if (i > 0) and (temp[i - 1] > 0):
            if (format == 'percent'):
                incr.append((temp[i] - temp[i - 1]) * 100 / temp[i - 1])
            else:
                incr.append((temp[i] - temp[i - 1]))
        else:
            incr.append(0)
    return incr


def get_deaths(df, st_name):
    ret_df = df[df['state'] == st_name]['deaths']
    return ret_df


def set_plot_attr(xlabel, ylabel):
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.xticks(rotation=90)

def moving_average(lst, days):
    curr_df = pd.DataFrame(lst)
    mv_avg = curr_df.rolling(window=days).mean()
    return mv_avg


def plot_data(df, state_list):
    marker = itertools.cycle(('x', '+', '.', 'o', '*'))
    for key in state_list:
        print(key)
        plt.figure(1)
        days = 14
        mv_avg = moving_average(incr_change(get_cases(df, states[key]),0), days)
        plt.bar(get_date(df, states[key]), incr_change(get_cases(df, states[key]),0), label=key)
        plt.plot(mv_avg, label=key+'_'+str(days)+'-day_mv_avg', color='firebrick')
        set_plot_attr('Dates', 'Increase in cases')
        fname = 'C:/Users/yukidave' + '/' + key + '_cases.png'
        plt.savefig(fname)
        plt.figure(2)
        plt.plot(get_date(df, states[key]),get_cases(df, states[key]), label=key, marker=next(marker))
        set_plot_attr('Dates', 'cases')
        plt.figure(3)
        plt.plot(get_date(df, states[key]), incr_change(get_deaths(df, states[key]),0), label=key, marker=next(marker))
        set_plot_attr('Dates', '% change in deaths')

    return plt


def plot_arima(df, state_list):
    for key in state_list:
        df_full = pd.DataFrame(incr_change(get_cases(df, states[key]),0))
        train_len = int(df_full.size)
        test_len = int(df_full.size - train_len)
        cases_df = df_full[0:train_len]
        cases_test = df_full[train_len:]
        result = adfuller(cases_df)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

        # Original Series
        fig, axes = plt.subplots(4, 2, sharex=True)
        axes[0, 0].plot(cases_df)
        axes[0, 0].set_title('Original Series')
        plot_acf(cases_df, ax=axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(cases_df.diff());
        axes[1, 0].set_title('1st Order Differencing')
        plot_acf(cases_df.diff().dropna(), ax=axes[1, 1])

        # 1st Differencing
        axes[2, 0].plot(cases_df.diff().diff());
        axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(cases_df.diff().diff().dropna(), ax=axes[2, 1])

        axes[3, 0].plot(cases_df.diff().diff().diff());
        axes[3, 0].set_title('3rd Order Differencing')
        plot_acf(cases_df.diff().diff().diff().dropna(), ax=axes[3, 1])

        # PACF plot of 1st differenced series -- Get AR terms
        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(cases_df.diff());
        axes[0].set_title('1st Differencing')
        axes[1].set(ylim=(0, 5))
        plot_pacf(cases_df.diff().dropna(), ax=axes[1])

        # ACF plot of 1st differenced series -- Get MA terms
        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(cases_df.diff());
        axes[0].set_title('1st Differencing')
        axes[1].set(ylim=(0, 5))
        plot_acf(cases_df.diff().dropna(), ax=axes[1])

        # 1,1,2 ARIMA Model
        model = ARIMA(cases_df.values, order=(2, 1, 1))
        model_fit = model.fit(disp=0)
        print (model_fit.summary())

        # Plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        fig, ax = plt.subplots(1, 2)
        residuals.plot(title="Residuals", ax=ax[0])
        residuals.plot(kind='kde', title='Density', ax=ax[1])

        # Actual vs Fitted
        model_fit.plot_predict(dynamic=False)

        plt.figure('Forecast')
        forecast,se,ci = model_fit.forecast(15)

        print(cases_df.size)
        idx = range(cases_df.size, cases_df.size + 15)
        fc_series = pd.Series(forecast, index=idx)
        lower_series = pd.Series(ci[:, 0], index=idx)
        upper_series = pd.Series(ci[:, 1], index=idx)
        plt.plot(get_date(df, states[key]),cases_df, label='actual')
        #plt.plot(cases_test, label='test_actual')
        plt.plot(fc_series, label='forecast')
        plt.fill_between(lower_series.index, lower_series, upper_series,
                         color='k', alpha=.15)
        plt.legend(loc='upper left', fontsize=8)
        set_plot_attr('Dates', 'Cases Per Day')
        #residuals.plot(kind='kde')
    return plt

def plot_auto_arima(df, state_list):
    for key in state_list:
        df_full = pd.DataFrame(incr_change(get_cases(df, states[key]),0))
        train_len = int(df_full.size)
        test_len = int(df_full.size - train_len)
        cases_df = df_full[0:train_len]
        cases_test = df_full[train_len:]
        model = pm.auto_arima(cases_df, start_p=1, start_q=1,
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=5, max_q=5, # maximum p and q
                              m=1,              # frequency of series
                              d=None,           # let model determine 'd'
                              seasonal=False,   # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
    print(model.summary())

def run():
    df = pd.read_csv('us-states.csv')
    state_list = ['MA']
    plot_data(df, state_list).show()
    #plot_arima(df, state_list).show()
    #plot_auto_arima(df, state_list)


run()
