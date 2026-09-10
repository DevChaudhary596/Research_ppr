"""Frozen CLIP ViT-B/32 backbone for unpooled patch and token representation extraction."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import CLIPModel


class FrozenCLIPBackbone(nn.Module):
    """Encapsulates pre-trained CLIP ViT-B/32 model with strictly frozen weights.
    
    Extracts:
        - Unpooled visual patch embeddings: (B, 49, 768)
        - Sequence word token embeddings: (B, 77, 768)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.clip = CLIPModel.from_pretrained(model_name)
        
        # Projection layer to align text token dimension (512) to unified visual patch dimension (768)
        text_hidden_dim = self.clip.config.text_config.hidden_size  # 512
        vision_hidden_dim = self.clip.config.vision_config.hidden_size  # 768
        self.text_projection = nn.Linear(text_hidden_dim, vision_hidden_dim)
        
        # Directive 1: Never unfreeze CLIP backbone encoders
        self.freeze_backbone()
        
        if device is not None:
            self.to(device)

    def freeze_backbone(self) -> None:
        """Strictly sets requires_grad = False for all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract unpooled visual patches and text token embeddings without gradient tracking.
        
        Args:
            pixel_values: (B, 3, 224, 224) normalized image tensor.
            input_ids: (B, 77) tokenized input IDs.
            attention_mask: (B, 77) attention mask.
            
        Returns:
            visual_patches: (B, 49, 768) unpooled spatial patch embeddings.
            text_tokens: (B, 77, 768) sequence token embeddings aligned to 768 dim.
        """
        with torch.no_grad():
            # Visual encoder forward pass
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                return_dict=True
            )
            # last_hidden_state: (B, 50, 768) -> index 0 is [CLS], 1:50 are 49 spatial patches
            visual_patches = vision_outputs.last_hidden_state[:, 1:, :]  # (B, 49, 768)

            # Text encoder forward pass
            text_outputs = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # (B, 77, 512) -> project to (B, 77, 768)
            raw_text_tokens = text_outputs.last_hidden_state  # (B, 77, 512)
            text_tokens = self.text_projection(raw_text_tokens)  # (B, 77, 768)

        return visual_patches, text_tokens
