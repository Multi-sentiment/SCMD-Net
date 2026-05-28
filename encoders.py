"""
Multi-modal Feature Encoders
- Text: DeBERTa-v3 base
- Visual: Lightweight Ma-Net style encoder
- Audio: Wav2Vec 2.0 base
"""

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Model


class TextEncoder(nn.Module):
    """DeBERTa-based text feature extractor."""

    def __init__(self, model_name: str = "microsoft/deberta-v3-base", 
                 output_dim: int = 256, freeze: bool = False):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.projector = nn.Linear(self.deberta.config.hidden_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

        if freeze:
            for param in self.deberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask
        Returns:
            features: (B, L, D) projected text features
            pooled: (B, D) [CLS] token feature
        """
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, hidden)
        pooled = hidden[:, 0, :]            # [CLS] token

        hidden = self.dropout(hidden)
        features = self.norm(self.projector(hidden))
        pooled = self.projector(pooled)

        return features, pooled


class VisualEncoder(nn.Module):
    """
    Ma-Net style visual encoder.
    Simplified implementation using a CNN-Transformer hybrid.
    In practice, replace with the actual Ma-Net implementation.
    """

    def __init__(self, input_dim: int = 2048, output_dim: int = 256,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Frame-level projection (simulating CNN feature extraction)
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Temporal modeling with Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, visual_frames: torch.Tensor, frame_mask: torch.Tensor = None):
        """
        Args:
            visual_frames: (B, T, input_dim) frame features
            frame_mask: (B, T) valid frame mask
        Returns:
            features: (B, T, output_dim) encoded visual features
        """
        # Frame-level encoding
        features = self.frame_encoder(visual_frames)  # (B, T, D)

        # Apply mask before temporal encoding
        if frame_mask is not None:
            features = features * frame_mask.unsqueeze(-1)

        # Temporal encoding
        features = self.temporal_encoder(features)  # (B, T, D)

        return features


class AudioEncoder(nn.Module):
    """Wav2Vec 2.0 based audio feature extractor."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h",
                 output_dim: int = 256, freeze: bool = False):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.projector = nn.Linear(self.wav2vec.config.hidden_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

        if freeze:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

    def forward(self, audio_values: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            audio_values: (B, T) raw audio waveform
            attention_mask: (B, T) audio attention mask
        Returns:
            features: (B, L, D) encoded audio features
        """
        outputs = self.wav2vec(audio_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, hidden)

        hidden = self.dropout(hidden)
        features = self.norm(self.projector(hidden))

        return features
