# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:57:04 2024

@author: kayla
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np




# Load the merged dataset
merged_df = pd.read_csv('Merged_Movie_Data.csv')


# Use 'Certificate_x' (from RT) or 'Certificate_y' (from IMDb) in the merged dataframe
merged_df['Certificate'] = merged_df['Certificate_x']  # Or merged_df['Certificate_y']

# Strip leading/trailing whitespace from the Certificate column
merged_df['Certificate'] = merged_df['Certificate'].str.strip()


#1. Pie Chart: Proportion of movies by Certificate with Legend
plt.figure(figsize=(8, 8))
certificate_counts = merged_df['Certificate'].value_counts()
certificate_counts.plot(kind='pie', autopct='%1.1f%%', labels=None, title="Proportion of Movie Certificates")
plt.ylabel('')  # Hide the y-label
plt.legend(labels=certificate_counts.index, title="Certificates", loc="best")
plt.show()

# 2. Bar Plot: Average Runtime by Certificate
certificate_runtime = merged_df.groupby('Certificate')['Runtime'].mean().reset_index()

plt.figure(figsize=(12, 6))
sb.barplot(data=certificate_runtime, x='Certificate', y='Runtime', palette='viridis')
plt.title('Average Movie Runtime by Certificate')
plt.xlabel('Certificate')
plt.ylabel('Average Runtime (minutes)')
plt.xticks(rotation=45)
plt.show()




# 3. Scatter Plot: Runtime vs RT Audience Score
plt.figure(figsize=(12, 6))
sb.scatterplot(data=merged_df, x='Runtime', y='RT Audience Score', alpha=0.6)
plt.title('Runtime vs RT Audience Score')
plt.xlabel('Runtime (minutes)')
plt.ylabel('RT Audience Score')
plt.show()

# 4. Heatmap: Correlation between Numerical Ratings
ratings_corr = merged_df[['RT Critics Score', 'RT Audience Score', 'iMDb Critics Score', 'iMDb Audience Reviews']].corr()

plt.figure(figsize=(8, 6))
sb.heatmap(ratings_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Ratings')
plt.show()
#%%

# # 5. Bar Plot: Movie Count by Certificate
# certificate_counts = merged_df['Certificate'].value_counts().reset_index()
# certificate_counts.columns = ['Certificate', 'Count']

# plt.figure(figsize=(12, 6))
# sb.barplot(data=certificate_counts, x='Certificate', y='Count', palette='Blues')
# plt.title('Movie Count by Certificate')
# plt.xlabel('Certificate')
# plt.ylabel('Movie Count')
# plt.xticks(rotation=45)
# plt.show()


# 6. Pair Plot: Multiple Ratings Comparison
ratings_df = merged_df[['RT Critics Score', 'RT Audience Score', 'iMDb Critics Score', 'iMDb Audience Reviews']]
sb.pairplot(ratings_df)
plt.title('Pairwise Comparison of Ratings')
plt.show()


# 8. Line Plot: RT Audience Score Over Years
yearly_ratings = merged_df.groupby('Year')['RT Audience Score'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(yearly_ratings['Year'], yearly_ratings['RT Audience Score'], marker='o')
plt.title('Average RT Audience Score Over Years')
plt.xlabel('Year')
plt.ylabel('Average RT Audience Score')
plt.show()


# # Select only the numeric columns for correlation calculation
# numeric_df = merged_df.select_dtypes(include=[np.number])
# corr_matrix = numeric_df.corr()
# plt.figure(figsize=(12, 6))
# sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()


# Count plot for categorical data
plt.figure(figsize=(10, 6))
sb.countplot(x='Certificate_y', data=merged_df, palette='magma')
plt.title('Distribution of Movie Certificates (y)')
plt.xlabel('Movie Certificate')
plt.ylabel('Count')
plt.show()
#%%
# Replace 'Critic_Score' and 'Audience_Score' with the actual column names containing numerical ratings
critic_mean = merged_df['RT Critics Score'].mean()
critic_median = merged_df['RT Critics Score'].median()
critic_std = merged_df['RT Critics Score'].std()

audience_mean = merged_df['iMDb Audience Reviews'].mean()
audience_median = merged_df['iMDb Audience Reviews'].median()
audience_std = merged_df['iMDb Audience Reviews'].std()

# Print the results
print(f"Critic Scores - Mean: {critic_mean:.2f}, Median: {critic_median:.2f}, Standard Deviation: {critic_std:.2f}")
print(f"Audience Scores - Mean: {audience_mean:.2f}, Median: {audience_median:.2f}, Standard Deviation: {audience_std:.2f}")

#%%
# Group by 'Genre' and calculate the mean for critic and audience scores
genre_avg_ratings = merged_df.groupby('Genre')[['RT Audience Score', 'RT Critics Score']].mean().reset_index()

# Plot the grouped bar chart
plt.figure(figsize=(14, 8))
bar_width = 0.35
x = range(len(genre_avg_ratings['Genre']))

# Bar positions
bar1_positions = [i - bar_width / 2 for i in x]
bar2_positions = [i + bar_width / 2 for i in x]

# Plot bars
bar1_plot = plt.bar(bar1_positions, genre_avg_ratings['RT Audience Score'], width=bar_width, label='RT Audience Score', color='blue')
bar2_plot = plt.bar(bar2_positions, genre_avg_ratings['RT Critics Score'], width=bar_width, label='RT Critics Score', color='orange')
plt.xticks(x, genre_avg_ratings['Genre'], rotation=45, ha='right')
plt.xlabel('Genre')
plt.ylabel('Average Score')
plt.title('Average RT Audience and Critics Score by Genre')
plt.legend()

# Annotate bars with their values
for bar in bar1_plot:
    height = bar.get_height()
    if not pd.isna(height):  # Skip NaN values
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}', ha='center', fontsize=9)

for bar in bar2_plot:
    height = bar.get_height()
    if not pd.isna(height):  # Skip NaN values
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()


#%%
# # Group by 'Genre' and calculate the mean for critic and audience scores
# genre_avg_ratings = merged_df.groupby('Genre')[['iMDb Critics Score', 'RT Critics Score']].mean().reset_index()

# # Plot the grouped bar chart
# plt.figure(figsize=(14, 8))
# bar_width = 0.35
# x = range(len(genre_avg_ratings['Genre']))

# # Bar positions
# bar1 = [i - bar_width/2 for i in x]
# bar2 = [i + bar_width/2 for i in x]

# # Plot bars
# plt.bar(bar1, genre_avg_ratings['iMDb Critics Score'], width=bar_width, label='iMDb Critics Score', color='blue')
# plt.bar(bar2, genre_avg_ratings['RT Critics Score'], width=bar_width, label='RT Critics Score', color='orange')

# # Add labels and title
# plt.xticks(x, genre_avg_ratings['Genre'], rotation=45, ha='right')
# plt.xlabel('Genre')
# plt.ylabel('Average Score')
# plt.title('Average RT and iMDb Audience and Crtics Score by Genre')
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()

