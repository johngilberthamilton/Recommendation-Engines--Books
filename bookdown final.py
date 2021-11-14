# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:49:18 2021

@author: HAMILJ37
"""

import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import statistics as stat
import sys
from sklearn.metrics.pairwise import cosine_similarity

x=1

##########################

ratings = pd.read_csv('C:/Users/hamilj37/OneDrive - Pfizer/Documents/book down/BX-Book-Ratings.csv', delimiter = ';', encoding = 'ISO-8859–1', )
books = pd.read_csv('C:/Users/hamilj37/OneDrive - Pfizer/Documents/book down/BX_Books.csv', delimiter = ';', encoding = 'ISO-8859–1', )
users = pd.read_csv('C:/Users/hamilj37/OneDrive - Pfizer/Documents/book down/BX-Users.csv', delimiter = ';', encoding = 'ISO-8859–1', )
#ratings_subset = pd.read_csv('C:/Users/hamilj37/OneDrive - Pfizer/Documents/book down/ratings_subset.csv')

ratings = ratings.rename(columns = {'User-ID':'user', 'ISBN':'isbn', 'Book-Rating':'rating'})
books = books.rename(columns = {'ISBN':'isbn', 'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication': 'year', 'Publisher':'publisher'})
ratings = ratings.reset_index()

ratings_subset = ratings_subset.astype({'rating':'int'})



# ratings = pd.concat([ratings, jgh_ratings])

x=2

######################

#adding in my own ratings

jgh_ratings = pd.read_csv('C:/Users/hamilj37/OneDrive - Pfizer/Documents/book down/jgh ratings 2.csv', dtype = 'object')
jgh_ratings_merge = jgh_ratings.merge(books, how = 'left', left_on = 'isbn', right_on = 'isbn')
x = 0
for lab, row in jgh_ratings_merge.iterrows():
    x=x+1
    find_str = jgh_ratings_merge.loc[lab, 'title']
    find = find_str.find('(')
    if find == -1:
        jgh_ratings_merge.loc[lab, 'title_clean'] = jgh_ratings_merge.loc[lab, 'title']
    else:
        jgh_ratings_merge.loc[lab, 'title_clean'] = jgh_ratings_merge.loc[lab, 'title'][0:find].rstrip()
    if x % 100 == 0: 
        print(x)
        
jgh_ratings_merge = jgh_ratings_merge.astype({'rating':'int'})

ratings_subset = pd.concat([ratings_subset, jgh_ratings_merge])

x=3

#####################################


# skip
# this takes a while- results are already saved in ratings_subset.csv

x = 0
for lab, row in books.iterrows():
    x=x+1
    find_str = books.loc[lab, 'title']
    find = find_str.find('(')
    if find == -1:
        books.loc[lab, 'title_clean'] = books.loc[lab, 'title']
    else:
        books.loc[lab, 'title_clean'] = books.loc[lab, 'title'][0:find].rstrip()
    if x % 100 == 0: 
        print(x)
        
#####################################

# skip- results saved in ratings_subset.csv
ratings_merge = ratings.merge(books, left_on = 'isbn', right_on = 'isbn', how = 'left')
ratings_merge = ratings_merge[ratings_merge.title.notnull()]

#ratings_merge = ratings_merge.astype({'title':'str'})

ratings_merge_subset = ratings_merge[ratings_merge.rating > 0]
ratings_pivot = ratings_merge_subset.groupby('title_clean').size().reset_index()
ratings_pivot.columns = ['title_clean', 'rating_count']
ratings_pivot_2 = ratings_pivot[ratings_pivot.rating_count > 2]
ratings_subset = ratings_merge_subset[ratings_merge_subset.title_clean.isin(ratings_pivot_2.title_clean)]

ratings_subset.to_csv('C:/Users/hamilj37/OneDrive - Pfizer/Documents/book down/ratings_subset.csv')

###############################################

#top rated books

ratings_pivot_3 = ratings_subset.groupby(by = 'title_clean', as_index = False).agg({'isbn':'nunique', 'rating':['count','mean','std']})
ratings_pivot_3.columns = ['title_clean', 'isbn_nunique', 'rating_count', 'rating_mean', 'rating_std']
ratings_pivot_3 = ratings_pivot_3.sort_values(by = 'rating_count', ascending = False)

#weighted average
m = ratings_pivot_3.rating_count.quantile(.8)
print(m)
C = ratings_pivot_3.rating_mean.mean()
ratings_pivot_3['rating_weighted'] = \
    ratings_pivot_3.rating_mean * (ratings_pivot_3.rating_count / (ratings_pivot_3.rating_count + m)) + \
        C * (m / (ratings_pivot_3.rating_count + m))

#it's definitely not perfect, but all the books at the top are great populist books, i'm satisfied for now. 
#this will be useful as a tiebreaker of sorts

x=4

###################################################

#find most similar users
# 1. make a matrix of ratings
# 2. fix NA values
# 3. cosine similarity / pearson / etc

#first- how much do we have to pair down the ratings?
ratings_subset = ratings_subset.astype({'rating':'int64', 'user': 'int64'})

print(ratings_subset.title_clean.nunique())
print(ratings_subset.user.nunique())

#ratings_subset_subset = ratings_subset.iloc[:200000,:]
#ratings_subset_subset_matrix = ratings_subset_subset.groupby(by = ['user', 'title_clean'], as_index = False).agg({'rating':'mean'})
ratings_subset.sort_values(by = ['user', 'title_clean'], axis = 0, inplace = True)
ratings_subset.reset_index(inplace = True)

ratings_subset_subset = ratings_subset.iloc[250000:,:]
ratings_matrix = ratings_subset_subset.pivot_table(index = 'user', columns = 'title_clean', values = 'rating')

ratings_matrix = ratings_matrix.fillna(0)

#######################################################


cosine_sim = cosine_similarity(ratings_matrix, ratings_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index = ratings_matrix.index, columns = ratings_matrix.index)
cosine_sim_df = cosine_sim_df.sort_values(by = 999999999, ascending = False)
winner_cosine = ratings_subset[ratings_subset.user == cosine_sim_df.iloc[1,:].name]
del(cosine_sim)
x=5

#it appears there's just not a ton of overlap. next steps-
#1. look at how many people have rated these different books- this will tell you if different methodologies might make an impact
#2. look at results of different methodologies and compare- moving to 0, doing pearson, etc.
#3. try rating other books

##########################################################

ratings_matrix_predict = pd.DataFrame(index = ratings_matrix.index, columns = ratings_matrix.columns) # creates empty data frame for scores
count_user = 0 # for timing while it's running
count_item = 0 # for timing while it's running
for i in ratings_matrix.columns: # for each book title, i is book titles
    user_subset = ratings_subset_subset[ratings_subset_subset.title_clean == i] # dataframe of user ratings for that book, not in matrix form
    count_item = count_item + 1 # for timing
    count_user = 0 # for timing
    for lab, row in ratings_matrix_predict.iterrows(): # going through each row in the blank score sheet, each row is a user's rank for everything, lab is a user
        cosine_subset = pd.DataFrame(cosine_sim_df[cosine_sim_df.index.isin(user_subset.user)][lab]) # find the users who have rated the book, and their 
                                                                                       # cosine similarities with the user being looped on
        sumproduct = (cosine_subset[lab]*user_subset.rating).sum()
        rating = sumproduct / cosine_subset[lab].sum()
        ratings_matrix_predict.loc[lab, i] = rating
        count_user = count_user + 1                              
        if count_user % 2900 == 0:
            print('item:', count_item, 'user:',count_user)