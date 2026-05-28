"""SCMD-Net: Semantic-Centric Multi-modal Decoupling Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .encoders import TextEncoder, VisualEncoder, AudioEncoder
from .context_transformer import ContextTransformer
from .lsc_stage import LSCStage
from .mbd_stage import MBDFusion


class SCMDNet(nn.Module):
    """
    SCMD-Net architecture with LSC and MBD stages.
    
    Stage 1 - LSC: Multi-modal encoding, context enhancement, Aura Vector extraction, dual-head optimization.
    Stage 2 - MBD: Semantic Key Calibration, Adaptive Residual Injection, concatenation + classification.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        model_cfg = config.get("model", {})
        lsc_cfg = config.get("lsc", {})
        mbd_cfg = config.get("mbd", {})
        
        text_dim = model_cfg.get("text_dim", 768)
        visual_dim = model_cfg.get("visual_dim", 512)
        audio_dim = model_cfg.get("audio_dim", 768)
        transformer_dim = model_cfg.get("transformer_dim", 256)
        num_layers = model_cfg.get("num_transformer_layers", 4)
        num_heads = model_cfg.get("num_heads", 8)
        dropout = model_cfg.get("dropout", 0.1)
        num_classes = model_cfg.get("num_classes", 3)
        
        aura_dim = lsc_cfg.get("aura_vector_dim", 64)
        gate_scale = mbd_cfg.get("gate_scale", 0.1)
        temperature = mbd_cfg.get("temperature", 1.0)

        self.text_encoder = TextEncoder(output_dim=transformer_dim, freeze=False)
        self.visual_encoder = VisualEncoder(
            input_dim=visual_dim, output_dim=transformer_dim,
            num_layers=2, num_heads=4, dropout=dropout
        )
        self.audio_encoder = AudioEncoder(output_dim=transformer_dim, freeze=False)

        self.text_transformer = ContextTransformer(
            dim=transformer_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout
        )
        self.visual_transformer = ContextTransformer(
            dim=transformer_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout
        )
        self.audio_transformer = ContextTransformer(
            dim=transformer_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout
        )

        self.text_lsc = LSCStage(input_dim=transformer_dim, aura_dim=aura_dim,
                                 num_classes=num_classes, dropout=dropout)
        self.visual_lsc = LSCStage(input_dim=transformer_dim, aura_dim=aura_dim,
                                   num_classes=num_classes, dropout=dropout)
        self.audio_lsc = LSCStage(input_dim=transformer_dim, aura_dim=aura_dim,
                                  num_classes=num_classes, dropout=dropout)

        self.lsc_stages = {
            "text": self.text_lsc,
            "visual": self.visual_lsc,
            "audio": self.audio_lsc,
        }

        self.mbd_fusion = MBDFusion(
            dim=transformer_dim, aura_dim=aura_dim, num_classes=num_classes,
            num_heads=num_heads, dropout=dropout,
            gate_scale=gate_scale, temperature=temperature,
        )

        self.task_weight = lsc_cfg.get("task_weight", 1.0)
        self.aura_weight = lsc_cfg.get("aura_weight", 0.5)
        self.fusion_loss_weight = mbd_cfg.get("fusion_loss_weight", 0.3)

    def encode_text(self, input_ids: torch.Tensor, 
                    attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_features, _ = self.text_encoder(input_ids, attention_mask)
        seq_features = self.text_transformer(seq_features, attention_mask)
        return seq_features, attention_mask

    def encode_visual(self, visual_frames: torch.Tensor,
                      frame_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_features = self.visual_encoder(visual_frames, frame_mask)
        seq_features = self.visual_transformer(seq_features, frame_mask)
        if frame_mask is None:
            frame_mask = torch.ones(visual_frames.shape[:2], device=visual_frames.device)
        return seq_features, frame_mask

    def encode_audio(self, audio_values: torch.Tensor,
                     audio_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_features = self.audio_encoder(audio_values, audio_mask)
        
        if audio_mask is not None:
            feature_len = seq_features.shape[1]
            original_len = audio_mask.shape[1]
            if feature_len != original_len:
                audio_mask = F.interpolate(
                    audio_mask.unsqueeze(1).float(), 
                    size=feature_len, mode='nearest'
                ).squeeze(1).bool()
        else:
            audio_mask = torch.ones(seq_features.shape[:2], device=audio_values.device)
            
        seq_features = self.audio_transformer(seq_features, audio_mask)
        return seq_features, audio_mask

    def lsc_forward(self, seq_features: torch.Tensor, mask: torch.Tensor,
                    labels: torch.Tensor, lsc_stage: LSCStage, modality_idx: int) -> dict:
        aura, task_logits, aura_logits = lsc_stage(seq_features, mask)
        task_loss, aura_loss = lsc_stage.compute_loss(aura, task_logits, aura_logits, labels, mask)
        weighted_loss = lsc_stage.get_weighted_loss(
            task_loss, aura_loss, modality_idx, self.task_weight, self.aura_weight
        )
        return {
            "aura": aura, "task_logits": task_logits, "aura_logits": aura_logits,
            "task_loss": task_loss, "aura_loss": aura_loss, "weighted_loss": weighted_loss,
        }

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        visual_frames: torch.Tensor,
        audio_values: torch.Tensor,
        visual_frame_mask: torch.Tensor = None,
        audio_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Full forward pass of SCMD-Net.
        
        Returns:
            Dictionary containing fused_logits, aura_vectors, lsc_results, 
            total_loss (if labels provided), and task_logits.
        """
        text_features, text_mask = self.encode_text(text_input_ids, text_attention_mask)
        visual_features, visual_mask = self.encode_visual(visual_frames, visual_frame_mask)
        audio_features, audio_mask = self.encode_audio(audio_values, audio_attention_mask)

        seq_features = {"text": text_features, "visual": visual_features, "audio": audio_features}
        seq_masks = {"text": text_mask, "visual": visual_mask, "audio": audio_mask}

        aura_vectors = {}
        lsc_results = {}
        task_logits = {}
        total_lsc_loss = torch.tensor(0.0, device=text_features.device, dtype=text_features.dtype)

        modality_order = ["text", "visual", "audio"]
        for idx, modality_name in enumerate(modality_order):
            lsc_stage = self.lsc_stages[modality_name]
            result = self.lsc_forward(
                seq_features[modality_name], seq_masks[modality_name],
                labels, lsc_stage, modality_idx=idx
            )
            aura_vectors[modality_name] = result["aura"]
            lsc_results[modality_name] = result
            task_logits[modality_name] = result["task_logits"]

            if labels is not None:
                total_lsc_loss += result["weighted_loss"]

        fused_logits, fused_features, fusion_loss = self.mbd_fusion(seq_features, aura_vectors, seq_masks)

        outputs = {
            "fused_logits": fused_logits,
            "aura_vectors": aura_vectors,
            "lsc_results": lsc_results,
            "task_logits": task_logits,
            "fused_features": fused_features,
        }

        if labels is not None:
            total_loss = total_lsc_loss + self.fusion_loss_weight * fusion_loss
            fused_ce_loss = F.cross_entropy(fused_logits, labels)
            total_loss += fused_ce_loss

            outputs["total_loss"] = total_loss
            outputs["fused_ce_loss"] = fused_ce_loss
            outputs["fusion_loss"] = fusion_loss
            outputs["total_lsc_loss"] = total_lsc_loss

        return outputs

    def predict(self, **inputs) -> torch.Tensor:
        """Inference-only forward pass. Returns class predictions."""
        with torch.no_grad():
            outputs = self.forward(**inputs)
            return outputs["fused_logits"].argmax(dim=-1)
