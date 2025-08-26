#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 

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

#create arrays for the interpolation
guy_pop = pd.to_numeric(population['Guyana'], errors='coerce').to_numpy()
germ_pop = pd.to_numeric(population['Germany'], errors='coerce').to_numpy()
pal_pop = pd.to_numeric(population['Palau'], errors='coerce').to_numpy()
years = np.linspace(1,64,64)

#filter down arrays to specific data points for the interpolation
guy = guy_pop[[0,21,31,43,52,63]]
guy_yrs = np.array([1,22,32,44,53,64])
germ = germ_pop[[0,14,25,43,51,63]]
germ_yrs = np.array([1,15,26,44,52,64])
pal = pal_pop[[0,14,20,44,53,63]]
pal_yrs = np.array([1,15,21,45,54,64])

#define matrices A for the countries
A_guy = np.vander(guy_yrs,increasing = True)
A_germ = np.vander(germ_yrs,increasing = True)
A_pal = np.vander(pal_yrs,increasing = True)

#Solve Ac = y for c
c_guy = np.linalg.solve(A_guy, guy)
c_germ = np.linalg.solve(A_germ, germ)
c_pal = np.linalg.solve(A_pal, pal)

#create new x arrays for the plot
x_guy = np.linspace(np.min(guy_yrs), np.max(guy_yrs), 10000)
x_germ = np.linspace(np.min(germ_yrs), np.max(germ_yrs), 10000)
x_pal = np.linspace(np.min(pal_yrs), np.max(pal_yrs), 10000)

#use polyval to standardize our p function for all vectors c
p_guy = np.polyval(np.flip(c_guy), x_guy)
p_germ = np.polyval(np.flip(c_germ), x_germ)
p_pal = np.polyval(np.flip(c_pal), x_pal)

#plot 1: Guyana Selected Points
plt.figure(1)
plt.plot(x_guy, p_guy, 'r', linewidth = 4, label = "Interpolent")
plt.plot(guy_yrs, guy, 'ko', linewidth = 5, label = "Data")
plt.title('Guyana Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()

#plot 2: Germany Selected Points
plt.figure(2)
plt.plot(x_germ, p_germ, 'r', linewidth = 4, label = "Interpolent")
plt.plot(germ_yrs, germ, 'ko', linewidth = 5, label = "Data")
plt.title('Germany Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population (In Tens of Millions)')
plt.legend()
plt.grid()
plt.show()

#plot 3: Palau Selected Points
plt.figure(3)
plt.plot(x_pal, p_pal, 'r', linewidth = 4, label = "Interpolent")
plt.plot(pal_yrs, pal, 'ko', linewidth = 5, label = "Data")
plt.title('Palau Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()

#plot 4: Guyana Entire Data Set
plt.figure(4)
plt.plot(x_guy, p_guy, 'r', linewidth = 4, label = "Interpolent")
plt.plot(years, guy_pop, 'ko', linewidth = 5, label = "Data")
plt.title('Guyana Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()

#plot 5: Germany Entire Data Set
plt.figure(5)
plt.plot(x_germ, p_germ, 'r', linewidth = 4, label = "Interpolent")
plt.plot(years, germ_pop, 'ko', linewidth = 5, label = "Data")
plt.title('Germany Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population (In Tens of Millions)')
plt.legend()
plt.grid()
plt.show()

#plot 6: Palau Entire Data Set
plt.figure(6)
plt.plot(x_pal, p_pal, 'r', linewidth = 4, label = "Interpolent")
plt.plot(years, pal_pop, 'ko', linewidth = 5, label = "Data")
plt.title('Palau Population Over the Years')
plt.xlabel('Year (Start: 1960, End: 2023)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()