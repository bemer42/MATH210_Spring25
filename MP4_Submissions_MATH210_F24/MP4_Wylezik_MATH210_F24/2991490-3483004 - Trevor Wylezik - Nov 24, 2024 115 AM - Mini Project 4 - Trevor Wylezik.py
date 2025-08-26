import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
from tabulate import tabulate

# MATH 210 Project 4


#%%# ---------- CLEANING THE ROTTEN TOMATES DATA SET ---------- #%%#

# Load the dataset:
RT_df = pd.read_csv('RottenTomatoes.csv') # Input file path/name

# Remove the justification in the Certificate column:
RT_df['Certificate'] = RT_df['Certificate'].str.replace(r' \(.*\)' ,'',regex=True)

# Remove the word "minutes" in Runtime column:
RT_df['Runtime'] = RT_df['Runtime'].str.replace(r'minutes','',regex=True)

# Convert all Runtimes to float type:
RT_df['Runtime'] = RT_df['Runtime'].astype(float)

# Remove any excess directors in the Director Column:
RT_df['Director '] = RT_df['Director '].str.split(',').str[0]

# Convert all Studio values
RT_df['Studio'] = RT_df['Studio'].astype(str)

# Convert RT Critics Score to float:
RT_df['RT Critics Score'] = RT_df['RT Critics Score'].str.replace(r'%','',regex=True)
RT_df['RT Critics Score'] = pd.to_numeric(RT_df['RT Critics Score'])

# Convert RT Audience Score to float
RT_df['RT Audience Score'] = RT_df['RT Audience Score'].str.replace(r'%','',regex=True)
RT_df['RT Audience Score'] = pd.to_numeric(RT_df['RT Audience Score'])

# Convert RT Audience Reviews to float
RT_df['RT Audience Reviews'] = RT_df['RT Audience Reviews'].str.replace(r'\,','',regex=True)
RT_df['RT Audience Reviews'] = pd.to_numeric(RT_df['RT Audience Reviews'])

RT_df.to_csv('CleanedRottenTomatoes.csv')


#%%# ---------- CLEANING THE iMDb DATA SET ---------- #%%#

# Load the dataset:
IMDB_df = pd.read_csv('iMDb.csv') # Input file path/name

# Delete select columns:
IMDB_df = IMDB_df.drop(['Poster', 'Cast', 'Description', 'Review Title', 'Review'], axis=1)

# Remove any excess directors in the Genres Column:
IMDB_df['Genre'] = IMDB_df['Genre'].str.split(',').str[0]

# Turn rating certificate systems over to U.S.
str_2_str = {
    'U': 'G',
    'UA': 'PG',
    'UA 7+': 'PG',
    'UA 13+': 'PG-13',
    'UA 16+': 'R',
    'A': 'R',
    'S': 'NC-17'
    }

IMDB_df['Certificate'] = IMDB_df['Certificate'].replace(str_2_str)

# Convert Multiple Columns to floats:
IMDB_df['iMDb Critics Score'] = pd.to_numeric(IMDB_df['iMDb Critics Score'])
IMDB_df['iMDb Metascore'] = pd.to_numeric(IMDB_df['iMDb Metascore'])

IMDB_df['iMDb Audience Reviews'] = IMDB_df['iMDb Audience Reviews'].str.replace(r'\,','',regex=True)
IMDB_df['iMDb Audience Reviews'] = pd.to_numeric(IMDB_df['iMDb Audience Reviews'])

IMDB_df['iMDb Critic Reviews'] = IMDB_df['iMDb Critic Reviews'].str.replace(r'\,','',regex=True)
IMDB_df['iMDb Critic Reviews'] = pd.to_numeric(IMDB_df['iMDb Critic Reviews'])

# Save to Cleaned file:
IMDB_df.to_csv('CleanediMDb.csv')



#%%# ---------- MERGING THE DATA SETS ---------- #%%#

# Load the dataset:
IMDB_df = pd.read_csv('CleanediMDb.csv') # Input file path/name
RT_df = pd.read_csv('CleanedRottenTomatoes.csv') # Input file path/name

# Remove duplicate columns (Allows us to merge cleanly)
IMDB_df = IMDB_df.drop(['Runtime', 'Director', 'Certificate', 'Unnamed: 0'], axis=1)
RT_df = RT_df.drop(['Unnamed: 0'], axis=1)

# Merge the datasets
Merged_df = pd.DataFrame.merge(IMDB_df, RT_df, on='Title')




#%%# ---------- DESCRIPTIVE STATISTICS OF RT AND IMDB ---------- #%%#


## Statistical Analysis on RT Critics Score:
    
# Print a header for the console:
print("\nROTTEN TOMATOES CRITICS SCORE:")

# Five Number Summary
RT_CS = RT_df['RT Critics Score']
RT_CS_5Num = [np.min(RT_CS), 
              np.percentile(RT_CS, 25), 
              np.median(RT_CS), 
              np.percentile(RT_CS, 75), 
              np.max(RT_CS)]
np.set_printoptions(legacy='1.25')
print(f"The Five Number Summary: {RT_CS_5Num}")


# Mean and Standard Deviation
RT_CS_Mean = np.round(np.mean(RT_CS),2)
RT_CS_SD = np.round(np.std(RT_CS),2)
print(f"The Mean and Standard Deviation: {RT_CS_Mean}, {RT_CS_SD}\n")


## Statistical Analysis on RT Audience Score:

# Print a header for the console:
print("\nROTTEN TOMATOES AUDIENCE SCORE:")

# Five Number Summary for RT Audience Score: 
RT_AS = RT_df['RT Audience Score']
RT_AS_5Num = [np.min(RT_AS), 
              np.percentile(RT_AS, 25), 
              np.median(RT_AS), 
              np.percentile(RT_AS, 75), 
              np.max(RT_AS)]
print(f"The Five Number Summary of the Rotten Tomatoes Audience Scores: {RT_AS_5Num}")

# Mean and Standard Deviation
RT_AS_Mean = np.round(np.mean(RT_AS),2)
RT_AS_SD = np.round(np.std(RT_AS),2)
print(f"The Mean and Standard Deviation: {RT_AS_Mean}, {RT_AS_SD}\n")


## Statistical Analysis on iMDb Critics Score:

# Print a header for the console:
print("\nIMDB CRITICS SCORE:")

# Five Number Summary for iMDb Critics Score: 
IMDB_CS = IMDB_df['iMDb Critics Score']*10
IMDB_CS = IMDB_CS.dropna()
IMDB_CS_5Num = [np.min(IMDB_CS), 
              np.nanpercentile(IMDB_CS, 25), 
              np.nanmedian(IMDB_CS), 
              np.nanpercentile(IMDB_CS, 75), 
              np.max(IMDB_CS)]
print(f"The Five Number Summary of the Rotten Tomatoes Audience Scores: {IMDB_CS_5Num}")

# Mean and Standard Deviation
IMDB_CS_Mean = np.round(np.mean(IMDB_CS),2)
IMDB_CS_SD = np.round(np.std(IMDB_CS),2)
print(f"The Mean and Standard Deviation: {IMDB_CS_Mean}, {IMDB_CS_SD}\n")


## Statistical Analysis on iMDb Metascore:

# Print a header for the console:
print("\nIMDB METASCORE:")

# Five Number Summary for iMDb MetaScore: 
IMDB_MS = IMDB_df['iMDb Metascore'].dropna()
IMDB_MS_5Num = [np.min(IMDB_MS), 
              np.nanpercentile(IMDB_MS, 25), 
              np.nanmedian(IMDB_MS), 
              np.nanpercentile(IMDB_MS, 75), 
              np.max(IMDB_MS)]
print(f"The Five Number Summary of the Rotten Tomatoes Audience Scores: {IMDB_MS_5Num}")

# Mean and Standard Deviation
IMDB_MS_Mean = np.round(np.mean(IMDB_MS),2)
IMDB_MS_SD = np.round(np.std(IMDB_MS),2)
print(f"The Mean and Standard Deviation: {IMDB_MS_Mean}, {IMDB_MS_SD}\n")

#%%# ---------- BOX PLOTS ---------- #%%#

# Set up subplot structures
fig, axs = plt.subplots(2, 2)
fig.suptitle('Box and Whisker Plots (Rottem Tomatoes and IMDB)', fontsize=18)
fig.tight_layout()

# RT Critic Score Box Plots
axs[0, 0].boxplot(RT_CS, vert=False, showmeans=True)
axs[0, 0].set_yticks([])
axs[0, 0].set_title(f"RT Critics Score (mean={RT_CS_Mean})")
axs[0, 0].grid()

# RT Audience Score Box Plots
axs[0, 1].boxplot(RT_AS, vert=False, showmeans=True)
axs[0, 1].set_yticks([])
axs[0, 1].set_title(f"RT Audience Score (mean={RT_AS_Mean})")
axs[0, 1].grid()

# RT Critic Score Box Plots
axs[1, 0].boxplot(IMDB_CS, vert=False, showmeans=True)
axs[1, 0].set_yticks([])
axs[1, 0].set_title(f"IMDB Critics Score (mean={IMDB_CS_Mean})")
axs[1, 0].grid()

# RT Meta Score Box Plots
axs[1, 1].boxplot(IMDB_MS, vert=False, showmeans=True)
axs[1, 1].set_yticks([])
axs[1, 1].set_title(f"IMDB Metascore (mean={IMDB_MS_Mean})")
axs[1, 1].grid()




#%%# ---------- PIE CHART FOR IMDB ---------- #%%#


## Pie Charts

# Create a dataframe that only contains the "Genre" column:
IMDB_Genres_df = IMDB_df['Genre']

# Set up the string converstion for uncommon genres:
Genres_2_Other = {
    'Documentary': 'Other',
    'Fantasy': 'Other',
    'Thriller': 'Other',
    'Mystery': 'Other',
    'Sci-Fi': 'Other',
    'Romance': 'Other',
    'Western': 'Other',
    'Musical': 'Other',
    'Film-Noir': 'Other',
    'Family': 'Other',
    'History': 'Other',
    'Music': 'Other',
    'Sport': 'Other',
    'War': 'Other'
    }

# Replace all uncommon genres with "other"
IMDB_Genres_df = IMDB_Genres_df.replace(Genres_2_Other)

# Count all genre categories
IMDB_Genre_Counts = IMDB_Genres_df.value_counts()

# Make the Pie Chart
IMDB_Genre_Counts.plot.pie(autopct='%1.1f%%', startangle=140)
plt.ylabel('')  # Optional: hide the y-label
plt.title('IMDB Movie Genres', fontsize=20)




#%%# ---------- STACKED HISTOGRAM FOR IMDB ---------- #%%#


# Convert the Year Column to int
IMDB_df['Year'] = IMDB_df['Year'].fillna(0).astype(int)

IMDB_GY_df = IMDB_df[IMDB_df['Year'] > 1999].replace(Genres_2_Other).groupby(['Year','Genre']).size().unstack(fill_value=0)

# Stacked Histogram
IMDB_GY_df.plot(kind='bar', stacked=True)
plt.grid(axis='y', linestyle='-', color = 'gray', linewidth = 1)
plt.title('Genre Distribution by Year', fontsize=25)
plt.xlabel('Year', size=20)
plt.ylabel('Count', size=20)
plt.xticks(rotation=90)
plt.gca().tick_params(labelsize=10)
plt.tight_layout()
plt.show()




#%%# ---------- BAR CHART FOR RT ---------- #%%#

# Create an array of the RT critics score and certificate columns
RT_Cert_Rate_df = RT_df[['Certificate','RT Critics Score']].groupby(['Certificate']).mean('RT Critics Score')
RT_Cert_Rate_df = RT_Cert_Rate_df.reindex([' G', ' PG', ' PG-13', ' R', ' NC17', ' NR'])

# Create a bar chart for the RT Critics Score based on certificate rating
RT_Cert_Rate_df.plot.bar()
plt.grid(axis='y', linestyle='-', color = 'gray', linewidth = 1)
plt.title('Average RT Critics Score (by rating)', fontsize=20)
plt.xlabel('Certificate', size=20)
plt.xticks(rotation=0)
plt.ylabel('Average Critics Score', size=20)
plt.gca().tick_params(labelsize=15)
plt.ylim(0,100)
plt.tight_layout()
plt.show()

# Create an array of the RT audience score and certificate columns
RT_Aud_Rate_df = RT_df[['Certificate','RT Audience Score']].groupby(['Certificate']).mean('RT Audience Score')
RT_Aud_Rate_df = RT_Aud_Rate_df.reindex([' G', ' PG', ' PG-13', ' R', ' NC17', ' NR'])

# Create a bar chart for the RT Audience Score based on certificate rating
RT_Aud_Rate_df.plot.bar()
plt.grid(axis='y', linestyle='-', color = 'gray', linewidth = 1)
plt.title('Average RT Audience Score (by rating)', fontsize=20)
plt.xlabel('Certificate', size=20)
plt.xticks(rotation=0)
plt.ylabel('Average Audience Score', size=20)
plt.gca().tick_params(labelsize=15)
plt.ylim(0,100)
plt.tight_layout()
plt.show()



#%%# ---------- AUDIENCE VS. CRITICS SCORES RT ---------- #%%#


## By Certification

# Create a new dataframe that isolate the certificate ratings and audience/critics scores
RT_Cert_Scores_df = RT_df[['Certificate', 'RT Critics Score', 'RT Audience Score']].groupby(['Certificate']).mean(['RT Critics Score', 'RT Audience Score'])

# Add a percent different column between audience and critics scores
RT_Cert_Scores_df['% Difference'] = ((RT_Cert_Scores_df['RT Audience Score'] - RT_Cert_Scores_df['RT Critics Score']) 
                                     / RT_Cert_Scores_df['RT Critics Score']) * 100

# Add the counts of the certificates for context
RT_Cert_Scores_df['Counts'] = RT_df['Certificate'].value_counts()

# Reindex (ordered by movie friendliness)
RT_Cert_Scores_df = RT_Cert_Scores_df.reindex([' G', ' PG', ' PG-13', ' R', ' NC17', ' NR'])

# Round the values (they look annoying currently haha)
RT_Cert_Scores_df['RT Critics Score'] = np.round(RT_Cert_Scores_df['RT Critics Score'], 2)
RT_Cert_Scores_df['RT Audience Score'] = np.round(RT_Cert_Scores_df['RT Audience Score'], 2)
RT_Cert_Scores_df['% Difference'] = np.round(RT_Cert_Scores_df['% Difference'], 2)

print('\nROTTEN TOMATOES AUDIENCE VS. CRITICS BY CERTIFICATION:')
print(f'{RT_Cert_Scores_df}\n\n')




#%%# ---------- AUDIENCE VS. CRITICS SCORES IMDB ---------- #%%#


## By Genre

# Create a new dataframe that isolate the genre and meta/critics scores
IMDB_Genre_Scores_df = IMDB_df[['Genre', 'iMDb Critics Score', 'iMDb Metascore']].groupby(['Genre']).mean(['iMDb Critics Score', 'iMDb Metascore'])

# Scales the critics scores to match the meta score
IMDB_Genre_Scores_df['iMDb Critics Score'] = IMDB_Genre_Scores_df['iMDb Critics Score']*10

# Add a percent different column between audience and critics scores
IMDB_Genre_Scores_df['% Difference'] = ((IMDB_Genre_Scores_df['iMDb Metascore'] - IMDB_Genre_Scores_df['iMDb Critics Score']) 
                                     / IMDB_Genre_Scores_df['iMDb Critics Score']) * 100

# Add the counts of the certificates for context
IMDB_Genre_Scores_df['Counts'] = IMDB_df['Genre'].value_counts()

# Sort the dataframe by the number of genres
IMDB_Genre_Scores_df = IMDB_Genre_Scores_df.sort_values(by='Counts', ascending = False)

# Round the values (they look annoying currently haha)
IMDB_Genre_Scores_df['iMDb Critics Score'] = np.round(IMDB_Genre_Scores_df['iMDb Critics Score'], 2)
IMDB_Genre_Scores_df['iMDb Metascore'] = np.round(IMDB_Genre_Scores_df['iMDb Metascore'], 2)
IMDB_Genre_Scores_df['% Difference'] = np.round(IMDB_Genre_Scores_df['% Difference'], 2)

print('IMDB METASCORE VS. CRITICS BY GENRE:')
print(f'{IMDB_Genre_Scores_df}\n\n')




#%%# ---------- BOX PLOTS OF MERGED DATASET ---------- #%%#

# Caluluate the mean of each value to display on the box plot
Merged_RT_Crit_Mean = np.round(np.mean(Merged_df['RT Critics Score']),2)
Merged_RT_Aud_Mean = np.round(np.mean(Merged_df['RT Audience Score']),2)
Merged_IMDB_Crit_Mean = np.round(np.mean(Merged_df['iMDb Critics Score']*10),2)
Merged_IMDB_Meta_Mean = np.round(np.mean(Merged_df['iMDb Metascore']),2)

# Set up subplot structures
fig, axs = plt.subplots(2, 2)
fig.suptitle('Box and Whisker Plots (Common Movies)', fontsize=18)
fig.tight_layout()

# RT Critic Score Box Plots
axs[0, 0].boxplot(Merged_df['RT Critics Score'], vert=False, showmeans=True)
axs[0, 0].set_yticks([])
axs[0, 0].set_title(f"RT Critics Score (mean={Merged_RT_Crit_Mean})")
axs[0, 0].grid()

# RT Audience Score Box Plots
axs[0, 1].boxplot(Merged_df['RT Audience Score'], vert=False, showmeans=True)
axs[0, 1].set_yticks([])
axs[0, 1].set_title(f"RT Audience Score (mean={Merged_RT_Aud_Mean})")
axs[0, 1].grid()

# RT Critic Score Box Plots
axs[1, 0].boxplot((Merged_df['iMDb Critics Score']*10).dropna(), vert=False, showmeans=True)
axs[1, 0].set_yticks([])
axs[1, 0].set_title(f"IMDB Critics Score (mean={Merged_IMDB_Crit_Mean})")
axs[1, 0].grid()

# RT Meta Score Box Plots
axs[1, 1].boxplot((Merged_df['iMDb Metascore']*10).dropna(), vert=False, showmeans=True)
axs[1, 1].set_yticks([])
axs[1, 1].set_title(f"IMDB Metascore (mean={Merged_IMDB_Meta_Mean})")
axs[1, 1].grid()



