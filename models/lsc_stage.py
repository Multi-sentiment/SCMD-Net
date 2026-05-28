"""LSC (Latent Semantic Clustering) Stage."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuraVectorExtractor(nn.Module):
    """Compress sequence features into a compact Aura Vector using mean pooling + FFN."""

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
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            masked_sum = (x * mask_expanded).sum(dim=1)
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1e-8)
            pooled = masked_sum / valid_counts
        else:
            x_t = x.transpose(1, 2)
            pooled = self.pool(x_t).squeeze(-1)

        return self.ffn(pooled)


class TaskHead(nn.Module):
    """Classification head operating on full sequence features with attention pooling."""

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
        weights = self.attention_pool(x)
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        alpha = F.softmax(weights, dim=1)
        context = (x * alpha).sum(dim=1)
        return self.classifier(context)


class AuraHead(nn.Module):
    """Classification head operating on compressed Aura Vector."""

    def __init__(self, aura_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(aura_dim, aura_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aura_dim, num_classes),
        )

    def forward(self, aura: torch.Tensor) -> torch.Tensor:
        return self.classifier(aura)


class LSCStage(nn.Module):
    """Latent Semantic Clustering stage with Aura extraction and dual-head optimization."""

    def __init__(self, input_dim: int, aura_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.aura_extractor = AuraVectorExtractor(input_dim, aura_dim, dropout)
        self.task_head = TaskHead(input_dim, num_classes, dropout)
        self.aura_head = AuraHead(aura_dim, num_classes, dropout)
        self.modality_weights = nn.Parameter(torch.ones(3))

    def forward(self, seq_features: torch.Tensor, mask: torch.Tensor = None):
        aura = self.aura_extractor(seq_features, mask)
        task_logits = self.task_head(seq_features, mask)
        aura_logits = self.aura_head(aura)
        return aura, task_logits, aura_logits

    def compute_loss(self, aura: torch.Tensor, task_logits, aura_logits, labels, mask=None):
        """Compute combined LSC loss: task loss + aura loss + contrastive loss."""
        task_loss = F.cross_entropy(task_logits, labels)
        aura_loss = F.cross_entropy(aura_logits, labels)
        aura_contrastive = self._contrastive_loss(aura, labels)
        total_aura_loss = aura_loss + aura_contrastive
        return task_loss, total_aura_loss

    def _contrastive_loss(self, aura: torch.Tensor, labels: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """NT-Xent style contrastive loss on aura vectors."""
        aura = F.normalize(aura, dim=-1)
        sim_matrix = aura @ aura.T / temperature

        labels_expanded = labels.unsqueeze(0)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        identity = torch.eye(positive_mask.size(0), device=positive_mask.device)
        positive_mask = positive_mask - identity
        pos_counts = positive_mask.sum(dim=1).clamp(min=1e-8)

        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * (1 - identity) + identity * 1e-9
        denom = exp_sim.sum(dim=1)
        log_prob = sim_matrix - torch.log(denom + 1e-8)
        loss = -(positive_mask * log_prob).sum(dim=1) / pos_counts
        return loss.mean()

    def get_weighted_loss(self, task_loss: torch.Tensor, aura_loss: torch.Tensor,
                          modality_idx: int, task_weight: float = 1.0, aura_weight: float = 0.5) -> torch.Tensor:
        w = self.modality_weights[modality_idx]
        w_normalized = w / self.modality_weights.sum()
        return w_normalized * (task_weight * task_loss + aura_weight * aura_loss)
