"""Multi-Head Cross-Attention Classifier and End-to-End Architecture."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from src.models.clip_backbone import FrozenCLIPBackbone


class CrossAttentionClassifier(nn.Module):
    """Computes fine-grained cross-attention between word tokens and visual patches.
    
    Architecture:
        - Multi-Head Cross-Attention (embed_dim=768, num_heads=8, batch_first=True)
          Q = Text Tokens (B, 77, 768)
          K = Visual Patches (B, 49, 768)
          V = Visual Patches (B, 49, 768)
        - Sequence Average Pooling -> (B, 768)
        - Projection MLP Head:
            Linear(768 -> 128) -> ReLU -> Dropout(0.2) -> Linear(128 -> 1)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-Head Attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # MLP Classification Head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        visual_patches: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing cross-attention and consistency score.
        
        Args:
            text_tokens: (B, 77, 768) sequence embeddings from CLIP text encoder.
            visual_patches: (B, 49, 768) patch embeddings from CLIP vision encoder.
            attention_mask: (B, 77) binary text attention mask (1 = valid, 0 = pad).
            return_attention: If True, returns attention weight matrix.
            
        Returns:
            Dict containing:
                - 'logits': (B, 1) unnormalized score for BCEWithLogitsLoss
                - 'consistency_score': (B, 1) probability S_c in [0.0, 1.0]
                - 'attention_weights': (B, 77, 49) cross-attention map (if return_attention=True)
        """
        # Cross-Attention: Query = Text, Key/Value = Visual
        # attn_output shape: (B, 77, 768), attn_weights shape: (B, 77, 49)
        attn_output, attn_weights = self.cross_attention(
            query=text_tokens,
            key=visual_patches,
            value=visual_patches,
            need_weights=return_attention,
            average_attn_weights=True
        )

        # Residual connection + LayerNorm
        attended_features = self.layer_norm(text_tokens + attn_output)

        # Average pooling across sequence dimension
        if attention_mask is not None:
            # Masked average pooling to ignore padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, 77, 1)
            sum_features = (attended_features * mask_expanded).sum(dim=1)  # (B, 768)
            mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-6)  # (B, 1)
            pooled = sum_features / mask_sum  # (B, 768)
        else:
            pooled = attended_features.mean(dim=1)  # (B, 768)

        # MLP projection head
        logits = self.mlp(pooled)  # (B, 1)
        consistency_score = torch.sigmoid(logits)  # (B, 1)

        result = {
            "logits": logits,
            "consistency_score": consistency_score,
            "pooled_features": pooled
        }

        if return_attention and attn_weights is not None:
            result["attention_weights"] = attn_weights

        return result


class MultimodalInconsistencyDetector(nn.Module):
    """End-to-End System combining frozen CLIP backbone with trainable Cross-Attention Head."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Backbone strictly frozen (~150M frozen params)
        self.backbone = FrozenCLIPBackbone(model_name=clip_model_name)
        
        # Trainable Cross-Attention Head (~2.4M trainable params)
        self.classifier = CrossAttentionClassifier(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """End-to-end forward pass from raw tensors to consistency score."""
        visual_patches, text_tokens = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return self.classifier(
            text_tokens=text_tokens,
            visual_patches=visual_patches,
            attention_mask=attention_mask,
            return_attention=return_attention
        )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, frozen_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
