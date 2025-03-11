import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import load_data, preprocess_data
from model import RecommendationModel
import pickle

# Load and preprocess data
print("Loading data...")
movies, ratings = load_data()
print(f"Data loaded: {len(ratings)} rows")

print("Preprocessing data...")
ratings, user_encoder, movie_encoder = preprocess_data(ratings)

# Prepare data for PyTorch
print("Preparing data for PyTorch...")
user_ids = torch.tensor(ratings['userId'].values, dtype=torch.long)
movie_ids = torch.tensor(ratings['movieId'].values, dtype=torch.long)
ratings = torch.tensor(ratings['rating'].values, dtype=torch.float32)

dataset = TensorDataset(user_ids, movie_ids, ratings)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Save encoders
with open('models/user_encoder.pkl', 'wb') as f:
    pickle.dump(user_encoder, f)

with open('models/movie_encoder.pkl', 'wb') as f:
    pickle.dump(movie_encoder, f)

# Initialize the model
print("Initializing the model...")
num_users = len(user_encoder.classes_)
num_movies = len(movie_encoder.classes_)
model = RecommendationModel(num_users, num_movies)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model
print("Starting training...")
for epoch in range(10):
    for i, batch in enumerate(dataloader):
        user_ids, movie_ids, ratings = batch
        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")
    print(f"Epoch {epoch + 1} completed, Loss: {loss.item()}")

# Save the model
print("Saving the model...")
torch.save(model.state_dict(), 'models/movie_recommendation_model.pth')
print("Script completed successfully.")
