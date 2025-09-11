"""Neural network models for music recommendation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MusicEmbeddingNet(nn.Module):
    """Neural network for embedding audio features."""
    
    def __init__(self, num_audio_features: int = 13, embedding_dim: int = 128):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            nn.Linear(num_audio_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the audio embedding network."""
        return F.normalize(self.audio_encoder(audio_features), dim=1)


class DeepCollaborativeFilter(nn.Module):
    """Deep collaborative filtering model for user-item interactions."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for collaborative filtering."""
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(x).squeeze()


class BradleyTerryModel(nn.Module):
    """Bradley-Terry model for pairwise preference learning."""
    
    def __init__(self, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.item_strengths = nn.Embedding(num_items, embedding_dim)
        self.strength_head = nn.Linear(embedding_dim, 1)
        
    def forward(self, item_a: torch.Tensor, item_b: torch.Tensor) -> torch.Tensor:
        """Forward pass for pairwise preference prediction."""
        strength_a = self.strength_head(self.item_strengths(item_a))
        strength_b = self.strength_head(self.item_strengths(item_b))
        return torch.sigmoid(strength_a - strength_b)
    
    def get_item_strengths(self) -> torch.Tensor:
        """Get learned preference strengths for all items."""
        with torch.no_grad():
            all_items = torch.arange(self.item_strengths.num_embeddings)
            strengths = self.strength_head(self.item_strengths(all_items))
            return strengths.squeeze()


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer."""
    
    def __init__(self, bradley_terry_model: BradleyTerryModel, learning_rate: float = 1e-4):
        self.bt_model = bradley_terry_model
        self.optimizer = torch.optim.Adam(self.bt_model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()
        
    def update_preferences(self, comparisons_batch: list) -> float:
        """
        Update model with batch of preference comparisons.
        
        Args:
            comparisons_batch: List of (item_a_id, item_b_id, preference) tuples
            preference: 1 if item_a preferred, 0 if item_b preferred
            
        Returns:
            Average loss for the batch
        """
        self.optimizer.zero_grad()
        
        total_loss = 0
        for item_a, item_b, preference in comparisons_batch:
            item_a_tensor = torch.tensor([item_a], dtype=torch.long)
            item_b_tensor = torch.tensor([item_b], dtype=torch.long)
            preference_tensor = torch.tensor([float(preference)], dtype=torch.float)
            
            prediction = self.bt_model(item_a_tensor, item_b_tensor)
            loss = self.loss_fn(prediction, preference_tensor)
            total_loss += loss
            
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item() / len(comparisons_batch)
    
    def get_item_strengths(self) -> torch.Tensor:
        """Get learned preference strengths for all items."""
        return self.bt_model.get_item_strengths()
