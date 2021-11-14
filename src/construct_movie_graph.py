import pandas as pd
import math 
from functools import reduce


def movie(x,y):
    i = single_rate[x]
    j = single_rate[y]
    ij = len(set(user_lists[x]).intersection(set(user_lists[y])))
    if ij == 0 : return 0 - math.log(i) - math.log(j) + math.log(d)
    else: return math.log(ij) - math.log(i) - math.log(j) + math.log(d)


def construct_movie_graph(dataset, threshold=3):
    global user_lists 
    global single_rate 
    global d
    
    user_lists = []
    movies = dataset['movie_id'].unique()

    for i in movies:
        user_lists_per_movies = dataset.loc[(dataset['movie_id']== i) & (dataset['score']>=threshold), 'user_id'].values
        user_lists.append(user_lists_per_movies)
        
    
    single_rate = [len(dataset[(dataset['movie_id']==i) & (dataset['score'] >= threshold)]) for i in range(0,499+1)]
    d = reduce(lambda x,y: x+y, single_rate)

    movies = dataset['movie_id'].unique()
    movie_comb = [[x, y] for x in movies for y in movies if y > x]
            
    movie_graph = pd.DataFrame(movie_comb, columns=['x','y'])
    movie_graph['weight'] = movie_graph.apply(lambda row: movie(row['x'],row['y']), axis=1)

    return movie_graph