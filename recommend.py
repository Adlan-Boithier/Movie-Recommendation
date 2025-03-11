import torch
import pickle
import numpy as np
import pandas as pd
from utils.utils import load_data
from model import RecommendationModel

# Load data
movies, ratings = load_data()

# Load encoders
with open('models/user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

with open('models/movie_encoder.pkl', 'rb') as f:
    movie_encoder = pickle.load(f)

# Filter movies to keep only those present in ratings.csv
movies = movies[movies['movieId'].isin(movie_encoder.classes_)]

# Load the trained model
num_users = len(user_encoder.classes_)
num_movies = len(movie_encoder.classes_)
model = RecommendationModel(num_users, num_movies)
model.load_state_dict(torch.load('models/movie_recommendation_model.pth'))
model.eval()

def recommend_movies(user_id, top_k=10):
    """
    Recommend movies for a given user.
    Args:
        user_id (int): ID of the user.
        top_k (int): Number of recommendations to return.
    Returns:
        recommended_movies (DataFrame): Top-k recommended movies.
    """
    if user_id not in user_encoder.classes_:
        raise ValueError(f"User ID {user_id} is not valid.")
    
    # Encode user ID
    user_encoded = user_encoder.transform([user_id])
    user_tensor = torch.tensor(np.full(len(movies), user_encoded), dtype=torch.long)

    # Encode movie IDs
    movie_ids_encoded = movie_encoder.transform(movies['movieId'].values)
    movie_ids = torch.tensor(movie_ids_encoded, dtype=torch.long)

    # Debug: Check tensor shapes
    print(f"User tensor shape: {user_tensor.shape}")
    print(f"Movie IDs shape: {movie_ids.shape}")

    with torch.no_grad():
        predictions = model(user_tensor, movie_ids)

    top_indices = predictions.argsort(descending=True)[:top_k]
    return movies.iloc[top_indices]

# Example: Recommend movies for user 1
if __name__ == "__main__":
    recommended_movies = recommend_movies(1)
    print(recommended_movies[['title', 'genres']])