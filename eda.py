import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_data

# Charger les données
movies, ratings = load_data()

# Afficher les premières lignes des données
print("Movies Data:")
print(movies.head())
print("\nRatings Data:")
print(ratings.head())

# Statistiques descriptives
print("\nDescriptive Statistics for Ratings:")
print(ratings['rating'].describe())

# Distribution des notes
plt.figure(figsize=(10, 6))
sns.histplot(ratings['rating'], bins=10, kde=True)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig("images/ratings_distribution.png")  # Sauvegarder le graphique
plt.close()  # Fermer la fenêtre

# Nombre de notes par utilisateur
user_rating_counts = ratings['userId'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(user_rating_counts, bins=50, kde=True)
plt.title("Number of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Frequency")
plt.savefig("images/ratings_per_user.png")  # Sauvegarder le graphique
plt.close()  # Fermer la fenêtre

# Nombre de notes par film
movie_rating_counts = ratings['movieId'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(movie_rating_counts, bins=50, kde=True)
plt.title("Number of Ratings per Movie")
plt.xlabel("Number of Ratings")
plt.ylabel("Frequency")
plt.savefig("images/ratings_per_movie.png")  # Sauvegarder le graphique
plt.close()  # Fermer la fenêtre

# Top 10 des films les plus notés
top_movies = ratings['movieId'].value_counts().head(10)
top_movies = pd.merge(top_movies, movies, left_index=True, right_on='movieId')
print("\nTop 10 Most Rated Movies:")
print(top_movies[['title', 'genres']])

# Top 10 des films les mieux notés
top_rated_movies = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10)
top_rated_movies = pd.merge(top_rated_movies, movies, left_index=True, right_on='movieId')
print("\nTop 10 Highest Rated Movies:")
print(top_rated_movies[['title', 'genres', 'rating']])