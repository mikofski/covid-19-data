import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_state_daily_dead(state_deaths):
    # %% cell 12
    # get a time series of the California deaths
    #california = deaths['California']
    #california.head()
    
    # to make this generic for any state, replace
    # "california" with "state_deaths" everywhere

    # %% cell 13
    # days with more than 1 death (non-zero deaths)
    nz_deaths = state_deaths[state_deaths > 0]

    # find day zero for California
    zero_day = nz_deaths.index[0]
    #print(f'day zero is {zero_day}')

    #nz_deaths  # display the list

    # %% cell 14
    # calculate the least squares fit in the form y = mx + b 
    # where x are the number of days since day zero
    days = pd.DatetimeIndex(nz_deaths.index) - np.datetime64(zero_day, 'D')

    # days is a timedelta in nanoseconds, convert to days as floats
    x = (days.values/3600/24/1e9).astype(float)  # days
    ndays = x.size  # how many days in day zero?

    # append a column of constants [1] to get a constant "b"
    A = np.concatenate([x.reshape([ndays, 1]), np.ones([ndays, 1])], axis=1)

    #x  # display the list of days as floats

    # %% cell 15
    # if death rate is exponential, then in logspace it's a straight line
    log_nz_deaths = np.log(nz_deaths.values)
    log_nz_deaths  # display list of non-zero State deaths in logspace

    # %% cell 16
    # calcuate least squares fit for y = mx + b,
    # y is the log of deaths, x is days since day zeroprint(f'RMSE: {np.exp(np.sqrt(result[1]/ndays))} deaths')
    result = np.linalg.lstsq(A, log_nz_deaths, rcond=None)

    # get the coefficients for the result
    m, b = result[0]
    #print(f'y = mx+b, m={m:g}, b={b:g}')
    #print(f'RMSE: {np.exp(np.sqrt(result[1]/ndays))} deaths')

    # for more information on NumPy Linear Algebra Least Squares see:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq

    # %% cell 17
    # try quadratic
    x_vec = x.reshape([ndays, 1])
    x2_vec = x_vec*x_vec
    const_vec = np.ones([ndays, 1])

    # append a column of constants [1] to get a constant "b"
    A2 = np.concatenate([x2_vec, x_vec, const_vec], axis=1)

    # calcuate least squares fit for y = mx + b,
    # y is the log of deaths, x is days since day zero
    result = np.linalg.lstsq(A2, log_nz_deaths, rcond=None)

    # get the coefficients for the result
    a = result[0]
    #print(f'y = a2*x^2 + a1*x + a0, a2={a[0]:g}, a1={a[1]:g}, a0={a[2]:g}')
    #print(f'RMSE: {np.exp(np.sqrt(result[1]/ndays))} deaths')

    # %% cell 18
    # day 24 fit
    #m24 = 0.217197
    #b24 = -0.360004
    #y24 = np.exp(x*m24 + b24)

    # %% cell 19
    # calculate the fitted deaths, and compare to the NY Times data
    y = np.exp(x*m + b)  # fit number of State deaths
    y2 = np.exp(np.dot(A2,a))
    # get the days as dates and create a Pandas DataFrame for convenience
    fit_days = pd.DatetimeIndex(pd.Timestamp(zero_day) + days)
    deaths_fit = pd.DataFrame(
        {'Line-fit': y, 'Quadratic-fit': y2, 'raw': nz_deaths.values}, index=fit_days)

    # %% cell 20
    #f, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    # make a log-plot
    #deaths_fit.plot(ax=ax[0], logy=True)
    #ax[0].set_title('Least Squares fit of State Deaths Since Zero Day')
    #ax[0].set_ylabel('Total State Deaths')

    # plot in linear space
    #deaths_fit.plot(ax=ax[1])
    #ax[1].set_ylim([0, 4000])
    #ax[1].set_ylabel('Total State Deaths')
    #f.tight_layout()

    # %% cell 21
    # find max
    x_max = -a[1] / 2 / a[0]
    #print(f'quadratic fit max at x = {x_max:g}')

    # %% cell 22
    # deaths per day
    daily_dead = np.diff(np.append(0, nz_deaths.values))

    # because some days have zero deaths, and log(0)-> Inf, then we can't use
    # least-squares, so instead we use a minimize from SciPy optimization library

    # First we define our objective function
    # ======================================
    # We'll use a Gaussian/Normal distribution (bell curve) because it looks similar to the SEIR model
    # WARNING: this is only a tutorial, and a Gaussian/Normal PDF is only used as an example, there is
    # no science to it

    # NOTE: according to some expert epidemiologists, the SEIR model would be more appropriate 
    # https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model
    # The Institute for Health Metrics and Evaluation (IMHE) at the University of Washington
    # (https://covid19.healthdata.org/) is using a SEIR model, see the section
    # "WHAT’S IN THE DEVELOPMENT PIPELINE FOR IHME COVID-19 PREDICTIONS"
    # http://www.healthdata.org/covid/updates

    def gaussfit(a, x):
        shape, location, scale = a
        return shape/(scale * np.sqrt(2*np.pi)) * np.exp(-((x-location)/scale)**2 / 2) 

    # NOTE: we could also use norm.pdf or norm.cdf from scipy.stats instead of writing it out
    # https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.norm.html

    # now we call minimize the RMSE of between the objective function: "gausefit"
    # and the actual values we're trying to fit "daily_dead"
    # RMSE = root-mean-square error
    res = minimize(
        lambda a: np.sqrt(np.mean((daily_dead - gaussfit(a, x))**2)),  # RMSE
        x0=[3600, 60, 14])                                              # initial guess
    #print(res)  # print results from minimize

    # get the coefficients from the results
    shape, location, scale = res.x

    # now fit the daily dead using the objective and the coefficients
    dead_fit = gaussfit((shape, location, scale), x)

    # %% cell 23
    # is it flattening?
    #f = plt.figure(figsize=(16, 10))
    #plt.bar(fit_days, daily_dead)
    #plt.plot(fit_days, dead_fit, 'r--')
    #plt.plot(fit_days, (1 + 2*res.fun/shape) * dead_fit, 'b--')
    #plt.title('State')
    #plt.ylabel('deaths per day')
    #plt.xlabel('day')
    #plt.legend(['fit', '2*RMSE', 'raw'])
    #f.tight_layout()

    # %% cell 24
    # Use a 5-day rolling average to smooth the spikes
    daily_dead_smooth = pd.DataFrame({'daily dead smooth': daily_dead}, index=fit_days).rolling('5D').mean()
    daily_dead_smooth['dead fit'] = dead_fit
    daily_dead_smooth['2*RMSE'] = (1 + 2*res.fun/shape) * dead_fit
    #daily_dead_smooth.plot(figsize=(16, 10))
    #plt.title('State')
    #plt.ylabel('deaths per day')

    fit_results = fit_days, daily_dead, dead_fit
    fit_params = shape, location, scale
    nz_data = nz_deaths, zero_day, days, x, ndays, log_nz_deaths

    return daily_dead_smooth, fit_results, fit_params, nz_data