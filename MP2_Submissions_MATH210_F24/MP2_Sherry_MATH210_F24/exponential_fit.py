#import libraries
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

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

#create arrays for the curve fitting
mad = pd.to_numeric(population['Madagascar'], errors='coerce').to_numpy()
eth = pd.to_numeric(population['Ethiopia'], errors='coerce').to_numpy()
ang = pd.to_numeric(population['Angola'], errors='coerce').to_numpy()
years = np.linspace(1,64,64)

#build an exponential version of the vandermonde matrix (same for each country)
A = np.vstack((np.ones(len(years)), years)).T

#define the normal equation for each country (A_norm is the same for each)
A_norm = A.T @ A
y_mad = A.T @ np.log(mad)
y_eth = A.T @ np.log(eth)
y_ang = A.T @ np.log(ang)

#solve Ac = y for c:
c_mad = np.linalg.solve(A_norm, y_mad)
c_eth = np.linalg.solve(A_norm, y_eth)
c_ang = np.linalg.solve(A_norm, y_ang)

#define a and b for each country
a_mad = np.exp(c_mad[0])
b_mad = c_mad[1]
a_eth = np.exp(c_eth[0])
b_eth = c_eth[1]
a_ang = np.exp(c_ang[0])
b_ang = c_ang[1]

#create a new x array for the plot and extrapolate 
x_plot = np.linspace(np.min(years), 1.3*np.max(years), 10000)

#plot 1: Madagascar
plt.figure(1)
plt.plot(x_plot, a_mad*np.exp(b_mad*x_plot), 'r', linewidth = 4, label = "Fitted Curve")
plt.plot(years, mad, 'ko', linewidth = 5, label = "Data")
plt.title('Madagascar Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population (In Tens of Millions)')
plt.legend()
plt.grid()
plt.show()

#plot 2: Ethiopia
plt.figure(2)
plt.plot(x_plot, a_eth*np.exp(b_eth*x_plot), 'r', linewidth = 4, label = "Fitted Curve")
plt.plot(years, eth, 'ko', linewidth = 5, label = "Data")
plt.title('Ethiopia Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population (In Hundreds of Millions)')
plt.legend()
plt.grid()
plt.show()

#plot 3: Angola
plt.figure(3)
plt.plot(x_plot, a_ang*np.exp(b_ang*x_plot), 'r', linewidth = 4, label = "Fitted Curve")
plt.plot(years, ang, 'ko', linewidth = 5, label = "Data")
plt.title('Angola Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population (In Tens of Millions)')
plt.legend()
plt.grid()
plt.show()

#calculate predicted values
predicted_mad = a_mad * np.exp(b_mad * years)
predicted_eth = a_eth * np.exp(b_eth * years)
predicted_ang = a_ang * np.exp(b_ang * years)

#calculate R^2 for each country
def calculate_r_squared(y_actual, y_predicted):
    ss_total = np.sum((y_actual - np.mean(y_actual))**2)
    ss_residual = np.sum((y_actual - y_predicted)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

r_squared_mad = calculate_r_squared(mad, predicted_mad)
r_squared_eth = calculate_r_squared(eth, predicted_eth)
r_squared_ang = calculate_r_squared(ang, predicted_ang)

#print R^2 values
print(f"R^2 for Madagascar: {r_squared_mad:.4f}")
print(f"R^2 for Ethiopia: {r_squared_eth:.4f}")
print(f"R^2 for Angola: {r_squared_ang:.4f}")