#############################################################################
# Program Title: 
# Creation Date: 
# Description: 
#
##### Imports
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re
from tabulate import tabulate
##### Functions

##### Parameters
#%% Cleaning datasets
rt = pd.read_csv('RottenTomatoes.csv')
rt['Certificate'] = rt['Certificate'].str.replace(r'\(.*\)', '', regex=True)
rt_to_drop = ['Runtime', 'Director ', 'RT Critic Reviews', 'RT Audience Reviews']
rt=rt.drop(columns=rt_to_drop)

db = pd.read_csv('iMDb.csv')
db_to_drop = ['Poster', 'iMDb Audience Reviews', 'iMDb Critic Reviews', 'Certificate', 
              'Review Title', 'Review']
db=db.drop(columns=db_to_drop)

clean_string = {
    ' PG ': 'PG', ' PG': 'PG', ' NR ': 'NR', ' NR': 'NR',
    ' R ': 'R', ' R': 'R', ' G': 'G', 'NC17': 'NC-17',
    ' NC17': 'NC-17', ' PG-13': 'PG-13', ' PG-13 ': 'PG-13',
    'Not Rated': 'NR', 'Unrated': 'NR', 'A': 'NC-17',
    'X': 'NC-17', 'U': 'G', 'Approved': 'Removable',
    '(Banned)': 'Removable', 'All': 'G', 'UA 16+': 'PG-13',
    'U/A 16+': 'PG-13', '13': 'PG-13', 'UA': 'G',
    'UA 7+': 'PG', '18+': 'NC-17', 'U/A': 'G',
    'UA 13+': 'PG-13', '12+': 'PG-13', '12': 'PG-13',
    '18': 'NC-17', '7': 'PG', 'GP': 'PG', 'M/PG': 'PG',
    '16': 'PG-13', '15+': 'PG-13', '16+': 'PG-13'
    }

removable = ['Approved', '(Banned)', 'nan']
rt['Certificate'] = rt['Certificate'].replace(clean_string)
movies = db.merge(rt, on='Title')
movies = movies[movies['Year'] != np.nan]

#Reclassify Values
movies['Genre'] = movies['Genre'].str.split(',').str.get(0)
movies['RT Audience Score'] = movies['RT Audience Score'].str.rstrip('%').astype(float)/100
movies['RT Critics Score'] = movies['RT Critics Score'].str.rstrip('%').astype(float)/100
movies['iMDb Metascore'] = movies['iMDb Metascore'].astype(float)/100
movies['iMDb Critics Score'] = movies['iMDb Critics Score'].astype(float)/10

#Remove empty values

movies.dropna(inplace=True)
movies.to_csv('Clean_Movies.csv', index=False)
#%% General Stats
I_audience = movies['iMDb Critics Score']
R_audience = movies['RT Audience Score']
I_critics = movies['iMDb Metascore']
R_critics = movies['RT Critics Score']

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
ax1.hist2d(x=I_audience, y=R_audience, bins=50, cmap='inferno')
ax1.set_ylabel('RT Audience Rating', fontsize=15)
ax1.set_xlabel('iMDb Audience Rating', fontsize=15)
ax2.hist2d(x=I_critics, y=R_critics, bins=50, cmap='inferno')
ax2.set_ylabel('RT Critics Rating', fontsize=15)
ax2.set_xlabel('iMDb Critics Rating', fontsize=15)
fig1.suptitle("iMDb v. Rotten Tomatoes: Heat Maps")
plt.gca().tick_params(labelsize=10)						#Set font size of axis intervals
plt.minorticks_on()	
plt.savefig('RT_v_db_heatmap.eps', format='eps')


fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12,5))
ax3.hist2d(x=R_audience, y=R_critics, bins=50, cmap='inferno')
ax3.set_xlabel('RT Audience Rating', fontsize=15)
ax3.set_ylabel('RT Critics Rating', fontsize=15)
ax4.hist2d(x=I_audience, y=I_critics, bins=50, cmap='inferno')
ax4.set_xlabel('iMDb Audience Rating', fontsize=15)
ax4.set_ylabel('iMDb Critics Rating', fontsize=15)
fig2.suptitle("iMDb & Rotten Tomatoes: Heat Maps")
plt.gca().tick_params(labelsize=10)						#Set font size of axis intervals
plt.minorticks_on()	
plt.savefig('Audience_v_critics_heatmap.eps', format='eps')

#%% Possibiity of Names leading to higher ratings
actor_descrip = pd.DataFrame(columns = movies.columns)

num_of_rows = movies.shape[0]

for row in range(1, num_of_rows):
    cast_list = str(movies.iloc[row]['Cast']).split(',')
    for name in cast_list:
        if name in str(movies.iloc[row]['Description']):
            actor_descrip.loc[len(actor_descrip.index)] = movies.iloc[row].values
            break
actor_ra = actor_descrip['RT Audience Score']
actor_rc = actor_descrip['RT Critics Score']


plt.figure(3)
plt.hist2d(x=actor_ra, y=actor_rc,bins=10, cmap='inferno')
plt.suptitle('Movies Including Actor Names in Description: Heatmap')
plt.xlabel('Rotten Tomatoes Audience Score')
plt.ylabel('Rotten Tomatoes Critics Score')
plt.savefig('Actor_names_heatmap.eps', format='eps')

#%% Pie Charts
top_aud = movies[movies['RT Audience Score'] >= 0.90]
count = top_aud['Director'].value_counts()
plt.figure(6)
count.plot.pie(autopct = '%1.1f%%', startangle=140)
plt.ylabel('')
plt.title('Directors with Ratings Higher than 0.90')
plt.savefig('top_director_pie.eps', format='eps')

bot_aud = movies[movies['RT Audience Score'] <= 0.10]
count = bot_aud['Director'].value_counts()
plt.figure(7)
count.plot.pie(autopct = '%1.1f%%', startangle=140)
plt.ylabel('')
plt.title('Directors with Ratings Lower than 0.10')
plt.savefig('bot_director_pie.eps', format='eps')

top_aud = movies[movies['RT Audience Score'] >= 0.90]
count = top_aud['Genre'].value_counts()
plt.figure(8)
count.plot.pie(autopct = '%1.1f%%', startangle=140)
plt.ylabel('')
plt.title('Genres with Ratings Higher than 0.90')
plt.savefig('top_genre_pie.eps', format='eps')

bot_aud = movies[movies['RT Audience Score'] <= 0.10]
count = bot_aud['Genre'].value_counts()
plt.figure(9)
count.plot.pie(autopct = '%1.1f%%', startangle=140)
plt.ylabel('')
plt.title('Genres with Ratings Lower than 0.10')
plt.savefig('bot_genre_pie.eps', format='eps')
#%% Box Plots
raud = movies['iMDb Metascore']

movies.boxplot(column='iMDb Metascore', by='Genre', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
plt.suptitle('')
plt.savefig('db_genre_box.eps', format='eps')
movies.boxplot(column='RT Audience Score', by='Genre', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
plt.suptitle('')
plt.savefig('rt_genre_box.eps', format='eps')
movies.boxplot(column='RT Audience Score', by='Certificate', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
plt.suptitle('')
plt.savefig('rt_genre_box.eps', format='eps')

#%% Bar Graph
certs = movies['Certificate'].value_counts()
top_certs = movies[movies['Year'] >= 2000]
top_certs = top_certs['Certificate'].value_counts()
bot_certs = movies[movies['Year'] < 2000]
bot_certs = bot_certs['Certificate'].value_counts()


plt.figure(10)
plt.bar(certs.index.values, certs.values)
plt.title('Distribution of Movies and Ratings')
plt.xlabel('Movie Ratings')
plt.ylabel('Number of Movies')
plt.savefig('bar_total_rating.eps', format='eps')

plt.figure(11)
plt.bar(top_certs.index.values, top_certs.values)
plt.title('Distribution of Movie Ratings Since 2000')
plt.xlabel('Movie Ratings')
plt.ylabel('Number of Movies')
plt.savefig('bar_recent_rating.eps', format='eps')

plt.figure(12)
plt.bar(bot_certs.index.values, bot_certs.values)
plt.title('Distribution of Movie Ratings Before 2000')
plt.xlabel('Movie Ratings')
plt.ylabel('Number of Movies')
plt.savefig('bar_older_rating.eps', format='eps')
plt.show()

movies['Certificate'].value_counts()




#Last Updated: 