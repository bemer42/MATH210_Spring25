#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:17:36 2024

@author: brooksemerick
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#load in dataframe
population = pd.read_csv("population_data.csv")

#drop unnecessary columns
population = population.drop(['Country Code', 'Indicator Name'], axis = 1)

#tranpose the dataframe to get it in the form that we want
population = population.T.reset_index()

#replace the column indices with the desired column headers
population.columns = population.iloc[0]
population = population[1:]
population.reset_index(drop=True, inplace=True)

#rename the 'Country Name' column to 'Year'
population.rename(columns={'Country Name': 'Year'}, inplace=True)

nau = pd.to_numeric(population['Nauru'], errors='coerce').to_numpy()
years = np.linspace(1,64,64)

# Normalize data to a trig type function
nau = nau/np.max(nau)
years = years/np.max(years)*2*np.pi

#function
def f(x,a,b,c,d):
    return  a + b*x + e*x**2 + c*np.sin(d*x)**2
    
#perform curve fitting
c_fit = sp.optimize.curve_fit(f, years, nau)
c_fit = c_fit[0]

#define x with more points to make a smooth curve
x_plot = np.linspace(np.min(years), np.max(years), 10000)

# Plotting the data and the fit
plt.figure(1)
plt.plot(years, nau, 'ko', linewidth = 5, label = "Data")
plt.plot(x_plot, f(x_plot, c_fit[0], c_fit[1], c_fit[2], c_fit[3]), 'r', linewidth = 4, label = "Fitted Curve")
plt.title('Nauru Population Over the Years')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid()
plt.legend()
plt.show()