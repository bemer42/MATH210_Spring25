#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#load in dataframe
population = pd.read_csv("C:\\Users\\Brady\\OneDrive\\Python Programs\\MATH 210\\Sherry_Project2\\population_data.csv")

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

#convert dataframe columns to arrays
nau = pd.to_numeric(population['Nauru'], errors='coerce').to_numpy()
years = np.linspace(1,64,64)

#normalize data to a trig type function
nau = nau/np.max(nau)
years = years/np.max(years)*2*np.pi

#define function to optimize 
def f(x,a,b,c,d,e,f):
    return a + b*x + c*np.cos(d*x)**2 + e*np.sin(f*x)
    
#perform curve fitting
c_fit = sp.optimize.curve_fit(f, years, nau)
c_fit = c_fit[0]

#define x with more points to make a smooth curve and extrapolate
x_plot = np.linspace(np.min(years), 2*np.max(years), 10000)

#plot the data and the fit
plt.figure(1)
plt.plot(years, nau, 'ko', linewidth = 5, label = "Data")
plt.plot(x_plot, f(x_plot, c_fit[0], c_fit[1], c_fit[2], c_fit[3], c_fit[4], c_fit[5]), 'r', linewidth = 4, label = "Fitted Curve")
plt.title('Nauru Population Over the Years')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid()
plt.legend()
plt.show()

#calculate predicted values
prediction = f(years, c_fit[0], c_fit[1], c_fit[2], c_fit[3], c_fit[4], c_fit[5])

#calculate R^2 for each country
def calculate_r_squared(y_actual, y_predicted):
    ss_total = np.sum((y_actual - np.mean(y_actual))**2)
    ss_residual = np.sum((y_actual - y_predicted)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

r_squared = calculate_r_squared(nau, prediction)

#print R^2 values
print(f"R^2 value: {r_squared:.4f}")