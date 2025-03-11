import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    """
    Load data from CSV files.
    Returns:
        movies (DataFrame): DataFrame containing movie data.
        ratings (DataFrame): DataFrame containing rating data.
    """
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    return movies, ratings

def preprocess_data(ratings):
    """
    Preprocess the data by encoding user and movie IDs.
    Args:
        ratings (DataFrame): DataFrame containing rating data.
    Returns:
        ratings (DataFrame): Preprocessed DataFrame.
        user_encoder (LabelEncoder): Encoder for user IDs.
        movie_encoder (LabelEncoder): Encoder for movie IDs.
    """
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
    ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])
    return ratings, user_encoder, movie_encoder