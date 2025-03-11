import torch
import torch.nn as nn

class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=50):
        """
        Initialize the recommendation model.
        Args:
            num_users (int): Number of unique users.
            num_movies (int): Number of unique movies.
            embedding_size (int): Size of the embedding vectors.
        """
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)

    def forward(self, user_ids, movie_ids):
        """
        Forward pass of the model.
        Args:
            user_ids (Tensor): Tensor of user IDs.
            movie_ids (Tensor): Tensor of movie IDs.
        Returns:
            predictions (Tensor): Predicted ratings.
        """
        user_embedded = self.user_embedding(user_ids)  # Shape: (batch_size, embedding_size)
        movie_embedded = self.movie_embedding(movie_ids)  # Shape: (batch_size, embedding_size)
        x = torch.cat([user_embedded, movie_embedded], dim=1)  # Shape: (batch_size, embedding_size * 2)
        return self.fc(x).squeeze()  # Shape: (batch_size,)