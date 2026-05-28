"""MBD (Multi-modal Bridging Decoupling) Stage."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticKeyCalibration(nn.Module):
    """Projects Aura Vectors to calibrate Key spaces using cross-modal semantic interaction."""

    def __init__(self, aura_dim: int, key_dim: int, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities

        self.aura_to_key = nn.ModuleList([
            nn.Sequential(nn.Linear(aura_dim, key_dim), nn.LayerNorm(key_dim))
            for _ in range(num_modalities)
        ])

        self.cross_modal_proj = nn.ModuleList([
            nn.Linear(aura_dim, key_dim) for _ in range(num_modalities)
        ])

        self.compatibility_scorer = nn.Sequential(
            nn.Linear(aura_dim * 2, aura_dim),
            nn.Tanh(),
            nn.Linear(aura_dim, 1),
        )

        self.consistency_scorer = nn.Sequential(
            nn.Linear(aura_dim, aura_dim // 2),
            nn.Tanh(),
            nn.Linear(aura_dim // 2, 1),
        )

        self.key_fusion = nn.Sequential(
            nn.Linear(key_dim * 2, key_dim),
            nn.Sigmoid(),
        )

    def forward(self, aura_vectors: dict, original_keys: dict, seq_masks: dict = None) -> dict:
        modality_names = list(aura_vectors.keys())
        calibrated_keys = {}

        for target_idx, target_name in enumerate(modality_names):
            target_aura = aura_vectors[target_name]
            target_keys = original_keys[target_name]
            B, L, _ = target_keys.shape

            target_key_guidance = self.aura_to_key[target_idx](target_aura)
            target_key_guidance = target_key_guidance.unsqueeze(1).expand(-1, L, -1)

            cross_modal_interaction = torch.zeros_like(target_keys)
            adaptive_weight_sum = torch.zeros(B, device=target_aura.device)

            for aux_idx, aux_name in enumerate(modality_names):
                if aux_name == target_name:
                    continue

                aux_aura = aura_vectors[aux_name]
                compat_input = torch.cat([target_aura, aux_aura], dim=-1)
                compatibility = F.softplus(self.compatibility_scorer(compat_input).squeeze(-1))
                consistency = F.softplus(self.consistency_scorer(aux_aura).squeeze(-1))
                weight = compatibility * consistency
                adaptive_weight_sum += weight

                aux_key_proj = self.cross_modal_proj[aux_idx](aux_aura)
                aux_key_proj = aux_key_proj.unsqueeze(1).expand(-1, L, -1)
                cross_modal_interaction += aux_key_proj * weight.unsqueeze(-1).unsqueeze(-1)

            adaptive_weight_sum = adaptive_weight_sum.clamp(min=1e-8)
            cross_modal_interaction = cross_modal_interaction / adaptive_weight_sum.unsqueeze(-1).unsqueeze(-1)

            fusion_input = torch.cat([target_keys, cross_modal_interaction], dim=-1)
            fusion_input = fusion_input.view(B * L, -1)
            fusion_gate = self.key_fusion(fusion_input).view(B, L, -1)
            
            calibrated = target_keys * fusion_gate + cross_modal_interaction * (1 - fusion_gate) + target_key_guidance * 0.1
            calibrated_keys[target_name] = calibrated

        return calibrated_keys


class AdaptiveResidualInjection(nn.Module):
    """Computes context via attention and modulates residual injection via Sigmoid gating."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, gate_scale: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gate_scale = gate_scale

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.residual_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim),
        )

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, original: torch.Tensor, calibrated_key: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = original.shape

        q = self.query_proj(original)
        k = self.key_proj(calibrated_key)
        v = self.value_proj(calibrated_key)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        context = self.output_proj(context)

        gate = torch.sigmoid(self.residual_gate(context)) * self.gate_scale
        fused = original + gate * context
        return self.layer_norm(fused)


class MBDFusion(nn.Module):
    """Multi-modal Bridging Decoupling module orchestrating SKC + ARI."""

    def __init__(self, dim: int, aura_dim: int, num_classes: int,
                 num_heads: int = 8, dropout: float = 0.1, 
                 gate_scale: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.num_modalities = 3
        self.dim = dim
        self.temperature = temperature

        self.skc = SemanticKeyCalibration(aura_dim, dim, self.num_modalities)

        self.ari_modules = nn.ModuleList([
            AdaptiveResidualInjection(dim, num_heads, dropout, gate_scale)
            for _ in range(self.num_modalities)
        ])

        self.fusion_head = nn.Sequential(
            nn.Linear(dim * self.num_modalities, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

        self.alignment_proj = nn.ModuleList([
            nn.Linear(dim, aura_dim) for _ in range(self.num_modalities)
        ])

    def forward(self, seq_features: dict, aura_vectors: dict, seq_masks: dict = None) -> tuple:
        modality_names = list(seq_features.keys())

        calibrated_keys = self.skc(aura_vectors, seq_features, seq_masks)

        fused_features = {}
        for i, modality_name in enumerate(modality_names):
            fused = self.ari_modules[i](
                seq_features[modality_name],
                calibrated_keys[modality_name],
                seq_masks.get(modality_name) if seq_masks else None
            )
            fused_features[modality_name] = fused

        pooled_features = []
        for modality_name in modality_names:
            feat = fused_features[modality_name]
            mask = seq_masks.get(modality_name) if seq_masks else None
            if mask is not None:
                mask_exp = mask.unsqueeze(-1)
                pooled = (feat * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-8)
            else:
                pooled = feat.mean(dim=1)
            pooled = F.layer_norm(pooled, [self.dim])
            pooled_features.append(pooled)

        concat_features = torch.cat(pooled_features, dim=-1)
        fused_logits = self.fusion_head(concat_features)

        fusion_loss = self._compute_alignment_loss(fused_features, aura_vectors)
        return fused_logits, fused_features, fusion_loss

    def _compute_alignment_loss(self, fused_features: dict, aura_vectors: dict) -> torch.Tensor:
        loss = 0.0
        count = 0
        modality_names = list(fused_features.keys())
        for i, modality_name in enumerate(modality_names):
            feat = fused_features[modality_name].mean(dim=1)
            proj = self.alignment_proj[i](feat)
            aura = aura_vectors[modality_name]
            loss += F.mse_loss(F.normalize(proj, dim=-1), F.normalize(aura, dim=-1))
            count += 1
        return loss / max(count, 1)
