"""
MBD (Multi-modal Bridging Decoupling) Stage
- Semantic Key Calibration: projects Aura Vectors to calibrate Key spaces
- Adaptive Residual Injection: sigmoid-gated residual fusion
- Cross-modal semantic interaction with adaptive weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticKeyCalibration(nn.Module):
    """
    Projects Aura Vectors into semantic guidance matrices to calibrate 
    each modality's Key space using cross-modal semantic interaction.
    
    For each target modality, uses auxiliary modalities' Aura Vectors to
    generate cross-modal semantic interaction terms. Computes semantic 
    compatibility between modalities and intra-modality consistency to
    derive adaptive weight coefficients.
    """

    def __init__(self, aura_dim: int, key_dim: int, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        self.aura_dim = aura_dim
        self.key_dim = key_dim

        # Project Aura Vector to Key space for each modality
        self.aura_to_key = nn.ModuleList([
            nn.Sequential(
                nn.Linear(aura_dim, key_dim),
                nn.LayerNorm(key_dim),
            ) for _ in range(num_modalities)
        ])

        # Cross-modal interaction projections
        self.cross_modal_proj = nn.ModuleList([
            nn.Linear(aura_dim, key_dim) for _ in range(num_modalities)
        ])

        # Semantic compatibility scorer
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(aura_dim * 2, aura_dim),
            nn.Tanh(),
            nn.Linear(aura_dim, 1),
        )

        # Intra-modality consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(aura_dim, aura_dim // 2),
            nn.Tanh(),
            nn.Linear(aura_dim // 2, 1),
        )

        # Key fusion gate
        self.key_fusion = nn.Sequential(
            nn.Linear(key_dim * 2, key_dim),
            nn.Sigmoid(),
        )

    def forward(self, aura_vectors: dict, original_keys: dict,
                seq_masks: dict = None) -> dict:
        """
        Calibrate Key spaces for each modality using cross-modal semantic guidance.
        
        Args:
            aura_vectors: dict of {modality_name: (B, aura_dim)} aura vectors
            original_keys: dict of {modality_name: (B, L, key_dim)} original key features
            seq_masks: dict of {modality_name: (B, L)} optional sequence masks
        Returns:
            calibrated_keys: dict of {modality_name: (B, L, key_dim)} unified emotional keys
        """
        modality_names = list(aura_vectors.keys())
        calibrated_keys = {}

        for target_idx, target_name in enumerate(modality_names):
            target_aura = aura_vectors[target_name]       # (B, aura_dim)
            target_keys = original_keys[target_name]       # (B, L, key_dim)
            B, L, _ = target_keys.shape

            # Project target's own aura to key space
            target_key_guidance = self.aura_to_key[target_idx](target_aura)  # (B, key_dim)
            target_key_guidance = target_key_guidance.unsqueeze(1).expand(-1, L, -1)  # (B, L, key_dim)

            # Compute cross-modal semantic interactions from auxiliary modalities
            cross_modal_interaction = torch.zeros_like(target_keys)  # (B, L, key_dim)
            adaptive_weight_sum = torch.zeros(B, device=target_aura.device)

            for aux_idx, aux_name in enumerate(modality_names):
                if aux_name == target_name:
                    continue

                aux_aura = aura_vectors[aux_name]  # (B, aura_dim)

                # 1. Semantic compatibility between target and auxiliary
                compat_input = torch.cat([target_aura, aux_aura], dim=-1)  # (B, 2*aura_dim)
                compatibility = self.compatibility_scorer(compat_input).squeeze(-1)  # (B,)
                compatibility = F.softplus(compatibility)  # Ensure non-negative

                # 2. Intra-modality consistency of auxiliary
                consistency = self.consistency_scorer(aux_aura).squeeze(-1)  # (B,)
                consistency = F.softplus(consistency)

                # Adaptive weight = compatibility * consistency
                weight = compatibility * consistency  # (B,)
                adaptive_weight_sum += weight

                # Cross-modal interaction term
                aux_key_proj = self.cross_modal_proj[aux_idx](aux_aura)  # (B, key_dim)
                aux_key_proj = aux_key_proj.unsqueeze(1).expand(-1, L, -1)  # (B, L, key_dim)
                cross_modal_interaction += aux_key_proj * weight.unsqueeze(-1).unsqueeze(-1)

            # Normalize by total weight
            adaptive_weight_sum = adaptive_weight_sum.clamp(min=1e-8)
            cross_modal_interaction = cross_modal_interaction / \
                adaptive_weight_sum.unsqueeze(-1).unsqueeze(-1)

            # Fuse original key with cross-modal interaction
            fusion_input = torch.cat([target_keys, cross_modal_interaction], dim=-1)  # (B, L, 2*key_dim)
            fusion_gate = self.key_fusion(fusion_input)  # (B, L, key_dim)
            
            # Add target's own guidance
            calibrated = target_keys * fusion_gate + \
                         cross_modal_interaction * (1 - fusion_gate) + \
                         target_key_guidance * 0.1  # Small residual from own guidance

            calibrated_keys[target_name] = calibrated

        return calibrated_keys


class AdaptiveResidualInjection(nn.Module):
    """
    Computes context representation via scaled dot-product attention,
    then dynamically modulates residual injection ratio via Sigmoid gating.
    Preserves original feature identity while resolving modality conflicts.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 gate_scale: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gate_scale = gate_scale

        # Q, K, V projections for cross-attention
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Sigmoid gate for residual injection ratio
        self.residual_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim),
        )

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, original: torch.Tensor, calibrated_key: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            original: (B, L, D) original modality features
            calibrated_key: (B, L, D) calibrated key features from SKC
            mask: (B, L) optional attention mask
        Returns:
            fused: (B, L, D) features after adaptive residual injection
        """
        B, L, D = original.shape

        # Compute Q from original, K and V from calibrated key
        q = self.query_proj(original)  # (B, L, D)
        k = self.key_proj(calibrated_key)  # (B, L, D)
        v = self.value_proj(calibrated_key)  # (B, L, D)

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, L, L)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0,
                float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Context representation
        context = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        context = self.output_proj(context)

        # Sigmoid-gated residual injection
        gate = torch.sigmoid(self.residual_gate(context)) * self.gate_scale  # (B, L, D)
        
        # Adaptive residual: preserve identity while injecting context
        fused = original + gate * context

        # Layer normalization
        fused = self.layer_norm(fused)

        return fused


class MBDFusion(nn.Module):
    """
    Multi-modal Bridging Decoupling module.
    Orchestrates Semantic Key Calibration + Adaptive Residual Injection
    for cross-modal fusion with modality conflict resolution.
    """

    def __init__(self, dim: int, aura_dim: int, num_classes: int,
                 num_heads: int = 8, dropout: float = 0.1, 
                 gate_scale: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.num_modalities = 3  # text, visual, audio
        self.dim = dim
        self.temperature = temperature

        # Semantic Key Calibration
        self.skc = SemanticKeyCalibration(aura_dim, dim, self.num_modalities)

        # Adaptive Residual Injection (one per modality)
        self.ari_modules = nn.ModuleList([
            AdaptiveResidualInjection(dim, num_heads, dropout, gate_scale)
            for _ in range(self.num_modalities)
        ])

        # Fusion classification head
        self.fusion_head = nn.Sequential(
            nn.Linear(dim * self.num_modalities, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

        # Fusion loss: encourages modality alignment
        self.alignment_proj = nn.ModuleList([
            nn.Linear(dim, aura_dim) for _ in range(self.num_modalities)
        ])

    def forward(self, seq_features: dict, aura_vectors: dict,
                seq_masks: dict = None) -> tuple:
        """
        Args:
            seq_features: dict of {modality_name: (B, L, D)} sequence features
            aura_vectors: dict of {modality_name: (B, aura_dim)} aura vectors
            seq_masks: dict of {modality_name: (B, L)} optional masks
        Returns:
            fused_logits: (B, num_classes) classification logits after fusion
            fused_features: dict of {modality_name: (B, L, D)} fused features
            fusion_loss: scalar alignment regularization loss
        """
        modality_names = list(seq_features.keys())

        # Step 1: Semantic Key Calibration
        calibrated_keys = self.skc(aura_vectors, seq_features, seq_masks)

        # Step 2: Adaptive Residual Injection per modality
        fused_features = {}
        for i, modality_name in enumerate(modality_names):
            fused = self.ari_modules[i](
                seq_features[modality_name],
                calibrated_keys[modality_name],
                seq_masks.get(modality_name) if seq_masks else None
            )
            fused_features[modality_name] = fused

        # Step 3: Layer norm + concat + classify
        pooled_features = []
        for modality_name in modality_names:
            feat = fused_features[modality_name]  # (B, L, D)
            mask = seq_masks.get(modality_name) if seq_masks else None
            if mask is not None:
                mask_exp = mask.unsqueeze(-1)
                pooled = (feat * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-8)
            else:
                pooled = feat.mean(dim=1)  # (B, D)
            pooled = F.layer_norm(pooled, [self.dim])
            pooled_features.append(pooled)

        # Concatenate all modality features
        concat_features = torch.cat(pooled_features, dim=-1)  # (B, D * num_modalities)
        fused_logits = self.fusion_head(concat_features)  # (B, num_classes)

        # Step 4: Compute fusion alignment loss
        fusion_loss = self._compute_alignment_loss(fused_features, aura_vectors)

        return fused_logits, fused_features, fusion_loss

    def _compute_alignment_loss(self, fused_features: dict, 
                                 aura_vectors: dict) -> torch.Tensor:
        """
        Regularization loss encouraging fused features to align with
        their respective aura vectors in the latent space.
        """
        loss = 0.0
        count = 0
        modality_names = list(fused_features.keys())
        for i, modality_name in enumerate(modality_names):
            # Mean-pool fused features
            feat = fused_features[modality_name].mean(dim=1)  # (B, D)
            
            # Project to aura space using modality-specific projection layer
            proj = self.alignment_proj[i](feat)  # (B, aura_dim)
            aura = aura_vectors[modality_name]  # (B, aura_dim)

            # MSE alignment loss
            loss += F.mse_loss(F.normalize(proj, dim=-1), 
                              F.normalize(aura, dim=-1))
            count += 1

        return loss / max(count, 1)
