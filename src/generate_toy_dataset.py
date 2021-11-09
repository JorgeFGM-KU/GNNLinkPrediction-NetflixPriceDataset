import numpy as np

destination_file_path = "data/toy_dataset/raw/ratings.txt"
n_users = 10000 # Number of users in dataset
n_movies = 500 # Number of movies in dataset
n_ratings_mean = 500 # Mean number of ratings per movie
n_ratings_std = np.round(n_ratings_mean*0.2).astype(np.int64) # Standard deviation of number of ratings per movie

n_ratings = np.round(np.random.normal(n_ratings_mean, n_ratings_std, n_movies)).astype(np.int64)

with open(destination_file_path, "w+") as file:
    for movie_id in range(n_movies):
        users = np.random.randint(n_users, size=n_ratings[movie_id])
        ratings = np.random.randint(low=1, high=6, size=n_ratings[movie_id])
        file.write(f"{movie_id}:\n")
        for i in range(n_ratings[movie_id]):
            file.write(f"{users[i]},{ratings[i]}\n")