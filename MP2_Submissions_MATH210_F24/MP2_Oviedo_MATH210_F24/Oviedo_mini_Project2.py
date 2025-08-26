# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:49:49 2024

@author: kayla
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define a function to handle fitting and plotting for multiple countries
def fit_gdp_data(country_name, years, gdp):
    # Linear interpolation
    f_linear = interp1d(years, gdp, kind='linear')
    gdp_2015 = f_linear(2015)
    print(f'{country_name} GDP in 2015 (Linear Interpolation): {gdp_2015:.2e}')

    # Plotting original data and linear interpolation
    plt.figure(figsize=(10, 6))  # Increase the plot size
    plt.plot(years, gdp, 'o', label=f'{country_name} Original Data')
    plt.plot(np.arange(2000, 2015.1, 0.1), f_linear(np.arange(2000, 2015.1, 0.1)), '-', label=f'{country_name} Linear Interpolation')
    plt.xlabel('year', fontsize=15)
    plt.ylabel('Production (Billions)', fontsize=15)
    plt.title(f'{country_name} Electricity Production from Renewable Resources (kWh)')
    plt.legend(loc='upper left')  # Reposition legend to upper left
    plt.show()

    # Normalize the years and GDP data for exponential fit
    years_normalized = years - 2000  # Shifting years to start from 0
    gdp_normalized = np.array(gdp) / 1e11  # Normalize GDP data

    # Define an exponential model: a * exp(b * x) + c
    def exp_model(x, a, b, c):
        return a * np.exp(b * x) + c

    # Provide initial guesses for exponential fit
    initial_guesses = [1.0, 0.1, 0.0]

    # Fit the exponential model with normalized data
    exp_params, _ = curve_fit(exp_model, years_normalized, gdp_normalized, p0=initial_guesses, maxfev=10000)

    # Predict GDP for 2020 (unnormalized result)
    gdp_2020_exp = exp_model(2020 - 2000, *exp_params) * 1e11
    print(f'{country_name} GDP in 2020 (Exponential Fit): {gdp_2020_exp:.2e}')

    # Plotting the exponential curve fit
    plt.figure(figsize=(10, 6))  # Increase the plot size
    plt.plot(years, gdp, 'o', label=f'{country_name} Original Data')
    plt.plot(np.arange(2000, 2025), exp_model(np.arange(2000, 2025) - 2000, *exp_params) * 1e11, '-', label=f'{country_name} Exponential Fit')
    plt.xlabel('year', fontsize=15)
    plt.ylabel('Production (Billions)', fontsize=15)
    plt.title(f'{country_name} Electricity Production from Renewable Resources (kWh)')
    plt.legend(loc='upper left')  # Reposition legend to upper left
    plt.show()

    # **Logarithmic Fit**
    # Define a logarithmic model: a * log(x) + b
    def log_model(x, a, b):
        return a * np.log(x) + b

    # Fit the logarithmic model to the data
    log_params, _ = curve_fit(log_model, years, gdp, maxfev=10000)

    # Predict GDP for the year 2020 using logarithmic fit
    gdp_2020_log = log_model(2020, *log_params)
    print(f'{country_name} GDP in 2020 (Logarithmic Fit): {gdp_2020_log:.2e}')

    # Plotting the logarithmic curve fit
    plt.figure(figsize=(10, 6))  # Increase the plot size
    plt.plot(years, gdp, 'o', label=f'{country_name} Original Data')
    plt.plot(np.arange(2000, 2025), log_model(np.arange(2000, 2025), *log_params), '-', label=f'{country_name} Logarithmic Fit')
    plt.xlabel('year', fontsize=15)
    plt.ylabel('Production (Billions)', fontsize=15)
    plt.title(f'{country_name} Electricity Production from Renewable Resources (kWh)')
    plt.legend(loc='upper left')  # Reposition legend to upper left
    plt.show()


    # Extracting the "b" value
    b_value = exp_params[1]
    print(f'{country_name} Exponential Fit Parameter b: {b_value:.2e}')  # Print the "b" value in scientific notation

    # Predict GDP for 2020 (unnormalized result)
    gdp_2020_exp = exp_model(2020 - 2000, *exp_params) * 1e11
    print(f'{country_name} GDP in 2020 (Exponential Fit): {gdp_2020_exp:.2e}')


# Years from 2000 to 2015
years = np.arange(2000, 2016)

# Data for multiple countries
gdp_data = {
    'China': [3174000000, 3334000000, 3467000000, 3640000000, 3936000000, 7434000000, 
             11096000000, 15647000000, 29803000000, 48011000000, 70255000000, 
             1.00573E+11, 1.3247E+11, 1.93807E+11, 2.2984E+11, 2.83851E+11],
    'Germany': [13743000000, 15162000000, 21353000000, 27869000000, 36538000000, 42867000000, 
               51630000000, 67151000000, 72806000000, 75827000000, 83856000000, 
               1.06102E+11, 1.21703E+11, 1.29368E+11, 1.42926E+11, 1.68389E+11],
    'United Kingdom': [4884000000, 5493000000, 6340000000, 7391000000, 9294000000, 12016000000, 
                       13513000000, 14612000000, 16706000000, 20017000000, 22220000000, 
                       28851000000, 35840000000, 48576000000, 58694000000, 77262000000],
    'India': [2964000000, 3983000000, 5178000000, 6862000000, 8547000000, 11025000000, 
              15353000000, 19258000000, 23259000000, 30514000000, 34065000000, 
              42495000000, 52150000000, 59079000000, 66931000000, 74143000000],
    'Japan': [13953000000, 13811000000, 14537000000, 15815000000, 16832000000, 18477000000, 
              19287000000, 20590000000, 20334000000, 20814000000, 37857000000, 
              39959000000, 42796000000, 51223000000, 66421000000, 80292000000]
}

# Loop through each country's data and apply the function
for country, gdp in gdp_data.items():
    fit_gdp_data(country, years, gdp)

