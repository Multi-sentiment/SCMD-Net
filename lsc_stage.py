"""
LSC (Latent Semantic Clustering) Stage
- Aura Vector extraction via global mean pooling + FFN
- Dual-head optimization: Task Head (sequence) + Aura Head (compressed)
- Modality-weighted loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AuraVectorExtractor(nn.Module):
    """
    Compress sequence features into a compact Aura Vector.
    Uses global mean pooling + Feed-Forward Network.
    """

    def __init__(self, input_dim: int, aura_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, aura_dim),
            nn.LayerNorm(aura_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) sequence features
            mask: (B, L) valid position mask (1 = valid)
        Returns:
            aura: (B, aura_dim) compact latent semantic center
        """
        # Global mean pooling (respecting mask)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
            masked_sum = (x * mask_expanded).sum(dim=1)  # (B, D)
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1e-8)  # (B, 1)
            pooled = masked_sum / valid_counts  # (B, D)
        else:
            # Simple mean over sequence dimension
            x_t = x.transpose(1, 2)  # (B, D, L)
            pooled = self.pool(x_t).squeeze(-1)  # (B, D)

        # Project to Aura Vector
        aura = self.ffn(pooled)
        return aura


class TaskHead(nn.Module):
    """
    Classification head operating on full sequence features.
    Preserves fine-grained detail information.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) sequence features
            mask: (B, L) attention mask
        Returns:
            logits: (B, num_classes) classification logits
        """
        # Attention-weighted pooling over sequence
        weights = self.attention_pool(x)  # (B, L, 1)
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        alpha = F.softmax(weights, dim=1)  # (B, L, 1)
        context = (x * alpha).sum(dim=1)  # (B, D)

        logits = self.classifier(context)
        return logits


class AuraHead(nn.Module):
    """
    Classification head operating on compressed Aura Vector.
    Supervises intra-class clustering.
    """

    def __init__(self, aura_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(aura_dim, aura_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aura_dim, num_classes),
        )

    def forward(self, aura: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aura: (B, aura_dim) compressed aura vectors
        Returns:
            logits: (B, num_classes) classification logits
        """
        return self.classifier(aura)


class LSCStage(nn.Module):
    """
    Latent Semantic Clustering stage.
    Extracts Aura Vectors and applies dual-head optimization.
    """

    def __init__(self, input_dim: int, aura_dim: int, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.aura_extractor = AuraVectorExtractor(input_dim, aura_dim, dropout)
        self.task_head = TaskHead(input_dim, num_classes, dropout)
        self.aura_head = AuraHead(aura_dim, num_classes, dropout)

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3))  # text, visual, audio

    def forward(self, seq_features: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            seq_features: (B, L, D) sequence features for one modality
            mask: (B, L) optional mask
        Returns:
            aura: (B, aura_dim) compressed aura vector
            task_logits: (B, num_classes) task head output
            aura_logits: (B, num_classes) aura head output
        """
        aura = self.aura_extractor(seq_features, mask)
        task_logits = self.task_head(seq_features, mask)
        aura_logits = self.aura_head(aura)
        return aura, task_logits, aura_logits

    def compute_loss(self, aura: torch.Tensor, task_logits, aura_logits, 
                     labels, mask=None):
        """
        Compute combined LSC loss: task loss + aura loss.
        Returns weighted loss using learnable modality weights.
        
        Args:
            aura: (B, aura_dim) pre-computed aura vector (avoid re-computation)
            task_logits: (B, num_classes) task head logits
            aura_logits: (B, num_classes) aura head logits
            labels: (B,) ground truth labels
            mask: optional attention mask
        """
        # Cross-entropy task loss
        task_loss = F.cross_entropy(task_logits, labels)
        
        # Aura clustering loss (cross-entropy on aura head output)
        aura_loss = F.cross_entropy(aura_logits, labels)

        # Also add contrastive loss on aura vectors for tighter clustering
        aura_contrastive = self._contrastive_loss(aura, labels)

        total_aura_loss = aura_loss + aura_contrastive

        return task_loss, total_aura_loss

    def _contrastive_loss(self, aura: torch.Tensor, labels: torch.Tensor,
                          temperature: float = 0.5) -> torch.Tensor:
        """
        NT-Xent style contrastive loss on aura vectors.
        Pulls same-class samples together, pushes different classes apart.
        """
        # Normalize aura vectors
        aura = F.normalize(aura, dim=-1)
        
        # Cosine similarity matrix
        sim_matrix = aura @ aura.T / temperature  # (B, B)

        # Positive mask (same class)
        labels_expanded = labels.unsqueeze(0)  # (1, B)
        positive_mask = (labels_expanded == labels_expanded.T).float()  # (B, B)
        
        # Remove self-similarity from positives
        identity = torch.eye(positive_mask.size(0), device=positive_mask.device)
        positive_mask = positive_mask - identity
        
        # Number of positives per sample
        pos_counts = positive_mask.sum(dim=1).clamp(min=1e-8)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        # Zero out diagonal to avoid self-similarity
        exp_sim = exp_sim * (1 - identity) + identity * 1e-9
        
        denom = exp_sim.sum(dim=1)  # (B,)
        log_prob = sim_matrix - torch.log(denom + 1e-8)
        loss = -(positive_mask * log_prob).sum(dim=1) / pos_counts
        return loss.mean()

    def get_weighted_loss(self, task_loss: torch.Tensor, aura_loss: torch.Tensor,
                          modality_idx: int, task_weight: float = 1.0,
                          aura_weight: float = 0.5) -> torch.Tensor:
        """Apply learnable modality weight to the loss."""
        w = self.modality_weights[modality_idx]
        w_normalized = w / self.modality_weights.sum()
        return w_normalized * (task_weight * task_loss + aura_weight * aura_loss)
