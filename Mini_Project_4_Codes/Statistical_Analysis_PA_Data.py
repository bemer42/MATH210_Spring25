#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 03:51:21 2024

@author: brooksemerick
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re
from tabulate import tabulate
# plt.close('all')

# This file performs data visulization and statistical tests on the data: 

# Load the datase:
# Full_df = pd.read_csv('Criminal_Data_Set_1976-to-2020.csv')
PA_df = pd.read_csv('Clean_PA_data.csv')
print(f"{np.shape(PA_df)}")

#%% Statistics and Visuals on Victim Data: 

V_age = PA_df['victim_1_age']

# Five Number Summary: 
mi = np.min(V_age)
q1 = np.percentile(V_age, 25)
M = np.median(V_age)
q3 = np.percentile(V_age, 75)
ma = np.max(V_age)

print(f"Five Number Summary: {mi, q1, M, q3, ma}")

# Mean and Standard Deviation: 
mean = np.mean(V_age)
st_dev = np.std(V_age)

print(f"Mean and St Dev: {mean, st_dev}")

# Plot a boxplot: 
plt.figure(1)
plt.boxplot(V_age, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
plt.title('Modified Boxplot with Outliers Marked', fontsize=20)
plt.xlabel('All Homicides', fontsize=16)
plt.ylabel('Victime Age', fontsize=16)
plt.grid(True, which='both')
plt.show()

# Plot a Relative Frequency Histgram of Victim Age Data: 
plt.figure(2)
V_age.plot(kind='hist', bins=20, density='True', edgecolor='black')
plt.title('Victim Age Histogram', fontsize=20)
plt.xlabel('Victime Age', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.grid(True, which='both')
plt.show()

# Plot a Relative Frequency Histgram of Victim Age Data: 
plt.figure(3)
V_age.plot(kind='hist', bins=20, density='True', edgecolor='black')
plt.title('Victim Age Histogram', fontsize=20)
plt.xlabel('Victime Age', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.grid(True, which='both')
plt.show()

# Plot the histogram
plt.figure(4)
plt.hist(V_age, bins=20, density=True, alpha=0.5, color='skyblue', edgecolor='black', label='Histogram')

# Add KDE plot
sb.kdeplot(V_age, color='darkblue', label='Density Curve')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with Density Curve')
plt.legend()

# Show plot
plt.show()

#%% Data visualizations by categories: 
    
# By sex:
PA_sex_df = PA_df[PA_df['victim_1_sex'] != 'unknown']

fig3, ax3 = plt.subplots(num=3)
PA_sex_df.boxplot(column='victim_1_age', by='victim_1_sex', grid=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
ax3.set_title('Victim Age Boxplots by Sex', fontsize=20)
ax3.set_xlabel('Sex', fontsize=16)
ax3.set_ylabel('Victim Age', fontsize=16)
plt.grid(True, which='both')
plt.suptitle('')
plt.show()

from scipy.stats import f_oneway

categories = [group['victim_1_age'].values for name, group in PA_df.groupby('victim_1_sex')]

# Perform one-way ANOVA
anova_result = f_oneway(*categories)
print("F-statistic:", anova_result.statistic)
print("p-value:", anova_result.pvalue)

#%%

# Counts by Relation:
PA_rel_df = PA_df[PA_df['victim_1_relation_to_offender_1'] != 'unknown']

fig4, ax4 = plt.subplots(num=4)
PA_rel_df['victim_1_relation_to_offender_1'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
ax4.set_title('Homocide Counts by Relation', fontsize=20)
ax4.set_xlabel('Relation', fontsize=16)
ax4.set_ylabel('Number of Homocides', fontsize=16)
plt.grid(True, which='both')
plt.suptitle('')
plt.show()

# By city:
PA_city_df = PA_df[PA_df['population_group'] != 'unknown']
   
fig5, ax5 = plt.subplots(num=5)
PA_city_df.boxplot(column='victim_1_age', by='population_group', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
ax5.set_title('Victim Age Boxplots by Population', fontsize=20)
ax5.set_xlabel('Sex', fontsize=16)
ax5.set_ylabel('Victim Age', fontsize=16)
plt.grid(True, which='both')
plt.suptitle('')
plt.show()


#%% 



# Create the two-way table
two_way_table = pd.crosstab(PA_df['victim_1_sex'], PA_df['victim_1_race'])

print(tabulate(two_way_table, headers='keys', tablefmt='pretty'))

#%% Bivariate Plot: 
    
offenders = PA_df['offender_1_age'].unique()

PA_df = PA_df[PA_df['offender_1_age'] != 'unknown']
PA_df = PA_df[PA_df['offender_1_age'] != 'nb']
PA_df = PA_df[PA_df['offender_1_age'] != 'bb']
PA_df.dropna(subset=['offender_1_age'], inplace=True) 
# Convert all entries in victim age column to floats:
PA_df['offender_1_age'] = pd.to_numeric(PA_df['offender_1_age'])

# Create a 2D KDE plot 
plt.figure(1)
sb.kdeplot(x=PA_df['victim_1_age'], y=PA_df['offender_1_age'], cmap="Blues", shade=True, thresh=0.05)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('2D KDE Plot with Seaborn')
plt.show()

plt.figure(2)
plt.hist2d(x=PA_df['victim_1_age'], y=PA_df['offender_1_age'], bins=30)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('2D KDE Plot with Seaborn')
plt.show()


#%%  Pie Chart


counts = PA_df['offender_1_race'].value_counts()

counts.plot.pie(autopct='%1.1f%%', startangle=140)
plt.ylabel('')  # Optional: hide the y-label
plt.title('Category Distribution')
plt.show()



#%% Means with error bars

grouped_data = PA_df.groupby('offender_1_weapon')['victim_1_age'].agg(['mean', 'std']).reset_index()

# Plot with error bars
plt.bar(grouped_data['offender_1_weapon'], grouped_data['mean'], yerr=grouped_data['std'],capsize=5, color=['skyblue', 'lightgreen', 'orange'])

# Add labels and title
plt.xlabel('Offender Weapon')
plt.ylabel('Mean Value')
plt.title('Mean Values with One Standard Deviation by Category')
plt.show()
