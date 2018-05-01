

import pandas as pd
from pandas import *
import scipy.stats as scp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from operator import itemgetter, attrgetter
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix as scm
from numpy import *
import ast
from scipy import stats

def load_movie_json(file_path):
    frame = pd.read_csv(file_path, dtype='unicode')
    
    json_columns = ['keywords', 'genres', 'production_companies', 'production_countries', 'spoken_languages']

    for column in json_columns:
    
        frame[column] = frame[column].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))
    
    return frame


def popularityGenre():
    data = load_movie_json('./tmdb_5000.csv')

    #Create a Dict to store genres
    genre_pop = {}
    
    #Create loop that runs through the shape of it
    for i in range(data.shape[0]):
        #Run through all of the genre JSON field
        for item in data['genres'][i]:
            #WITHIN JSON find the KEY NAME and make sure the POPULARITY is not NOT A NUMBER
            if 'name' in item and data.iloc[i]['popularity'] is not np.nan:
                
                #ASSIGN the GENRE first to a
                a = item['name']
                #ASSIGN to b
                b = float(data.iloc[i]['popularity'])
                
                #If that genre exists in genre_pop then add the popularity and count
                if a in genre_pop:
                    genre_pop[a]['popularity'] += b 
                    genre_pop[a]['count'] += 1
                #else create an entry with it
                else:
                    genre_pop[a] = {}
                    genre_pop[a]['genre'] = a
                    genre_pop[a]['popularity'] = b
                    genre_pop[a]['count'] = 0
    
    #Create the average/mean of the popularity                
    for i in genre_pop: 
        genre_pop[i]['popularity']/=genre_pop[i]['count']
    
    #for key, value in sorted(genre_pop.iteritems(), key=lambda (k,v): (v,k)):
        #print "%s: %s" % (key, value)
    
    #genretop = sorted(genre_pop.items(), key=itemgetter(1), reverse=False)
    
    print(genre_pop['Mystery']['popularity'])
    print(genretop.Foreign)
    #print(type(genre_pop))
    
    
#UNCOMMENT HERE TO RUN    
popularityGenre ()


def topActors():
    
    actors = {}
    
    data = pd.read_csv('./tmdb_5000.csv')
    #data = data.head(30)
    
    for index, row in data['cast'].iteritems():
        actorrow = ast.literal_eval(row)
        for i in actorrow:    
            if actorrow[i] in actors:
                actors[actorrow[i]]+= int(data['revenue'][index])  
            else:
                actors[actorrow[i]] = int(data['revenue'][index]) 
                        
    #data = data[['cast', 'revenue']].values.reshape(-1,2)
    topactors = sorted(actors.items(), key= lambda t: t[1], reverse=True)
    
    topactors = topactors[0:50]
    data['actorFact'] = 0
    count = 0
    for key, value in topactors:
        for index, row in data['cast'].iteritems():
            if key in row:
                data.loc[index,'actorFact'] = 1 
                
    print(data['actorFact'])

#topActors()