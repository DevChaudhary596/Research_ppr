"""Data preprocessing utilities for CLIP vision and text backbones."""

from typing import List, Union, Dict, Any
from PIL import Image
import torch
from transformers import CLIPProcessor, AutoProcessor


class CLIPDataPreprocessor:
    """Preprocesses images and text sequences for CLIP ViT-B/32 backbone.
    
    Transforms images into normalized (3, 224, 224) tensors and captions
    into tokenized IDs of max sequence length 77.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_text_len: int = 77,
        image_size: int = 224,
    ):
        self.model_name = model_name
        self.max_text_len = max_text_len
        self.image_size = image_size
        
        # Load HuggingFace CLIP processor
        self.processor = AutoProcessor.from_pretrained(model_name)

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocesses a single image (file path or PIL Image) into a normalized tensor.
        
        Args:
            image: File path or PIL Image instance.
            
        Returns:
            torch.Tensor: Normalized image tensor of shape (3, 224, 224).
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        # Squeeze batch dimension -> (3, 224, 224)
        return inputs["pixel_values"].squeeze(0)

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenizes text caption to fixed maximum sequence length (77).
        
        Args:
            text: Raw input caption string.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'input_ids' and 'attention_mask'
                                     of shape (77,).
        """
        inputs = self.processor(
            text=text,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }

    def __call__(
        self,
        image: Union[str, Image.Image],
        text: str
    ) -> Dict[str, torch.Tensor]:
        """Preprocess both image and text simultaneously.
        
        Args:
            image: Image path or PIL Image.
            text: Text caption string.
            
        Returns:
            Dict[str, torch.Tensor] containing 'pixel_values', 'input_ids', and 'attention_mask'.
        """
        pixel_values = self.preprocess_image(image)
        text_inputs = self.preprocess_text(text)
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }
