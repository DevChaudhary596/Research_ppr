"""Dataset and DataLoader module for NewsCLIPpings multimodal pairs."""

import os
import io
import random
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocess import CLIPDataPreprocessor


class NewsDataset(Dataset):
    """PyTorch Dataset for multimodal news pairs (image + caption).
    
    Each sample returns:
        - pixel_values: (3, 224, 224) float tensor
        - input_ids: (77,) int tensor
        - attention_mask: (77,) int tensor
        - label: (1,) float tensor (1.0 = Authentic/Pristine, 0.0 = Out-of-Context/Falsified)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        preprocessor: Optional[CLIPDataPreprocessor] = None,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        Args:
            samples: List of dictionaries with keys:
                     - 'image': PIL Image, image filepath, or raw bytes
                     - 'caption': string caption
                     - 'falsified': int/bool (0/False for authentic, 1/True for out-of-context)
            preprocessor: Instance of CLIPDataPreprocessor.
            model_name: Hugging Face model identifier for CLIP backbone.
        """
        self.samples = samples
        self.preprocessor = preprocessor or CLIPDataPreprocessor(model_name=model_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_data = sample["image"]
        caption = sample["caption"]
        
        # Consistency label: 1.0 = Pristine / Authentic, 0.0 = Falsified / Out-of-Context
        # falsified == 0 -> label = 1.0; falsified == 1 -> label = 0.0
        falsified = sample.get("falsified", 0)
        consistency_label = 0.0 if bool(falsified) else 1.0

        # Handle various image formats
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, str):
            if os.path.exists(image_data):
                image = Image.open(image_data).convert("RGB")
            else:
                # Fallback solid color image if path not found
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Preprocess visual and textual inputs
        processed = self.preprocessor(image=image, text=caption)
        
        return {
            "pixel_values": processed["pixel_values"],
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "label": torch.tensor([consistency_label], dtype=torch.float32),
        }


def generate_benchmark_samples(
    num_samples: int = 100,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generates synthetic balanced multimodal samples for local testing and pipeline validation.
    
    Creates authentic and out-of-context pairs with diverse color patches and news captions.
    """
    random.seed(seed)
    authentic_captions = [
        "Crowds gather at London's Trafalgar Square during a climate change demonstration.",
        "President addresses foreign diplomats during bilateral economic summit.",
        "Wildfire response teams combat spreading flames in northern California forests.",
        "Space shuttle prepares for launch at Kennedy Space Center launch pad.",
        "Athletes compete in the international championship marathon under heavy rain.",
        "Medical researchers review diagnostic data in university biomedical laboratory.",
        "Engineers inspect structural integrity of suspension bridge after earthquake.",
        "Floodwaters submerge commercial district streets following tropical storm.",
    ]

    mismatched_captions = [
        "Celebration fireworks light up Sydney harbor on New Year's Eve.",
        "Military convoy navigates desert dunes during tactical training exercise.",
        "Volcano erupts spewing massive ash plumes over island villages.",
        "Subway commuters evacuate metro station during electrical power outage.",
        "Protesters march outside European Parliament demanding judicial reforms.",
        "Rescue crews search avalanche debris in Swiss alpine resort.",
    ]

    palette = [
        (40, 100, 200), (220, 50, 40), (50, 180, 80),
        (230, 160, 20), (120, 60, 180), (70, 70, 70)
    ]

    samples = []
    for i in range(num_samples):
        # 50% authentic (falsified=0), 50% out-of-context (falsified=1)
        is_falsified = (i % 2 == 1)
        color = palette[i % len(palette)]
        img = Image.new("RGB", (224, 224), color=color)

        if not is_falsified:
            caption = authentic_captions[i % len(authentic_captions)]
        else:
            caption = mismatched_captions[i % len(mismatched_captions)]

        samples.append({
            "image": img,
            "caption": caption,
            "falsified": 1 if is_falsified else 0,
        })

    return samples


def stream_news_clippings_subset(
    split: str = "train",
    max_samples: int = 3500,
    dataset_name: str = "g-luo/news-clippings",
    streaming: bool = True,
    fallback_benchmark: bool = True
) -> List[Dict[str, Any]]:
    """Streams a balanced subset from Hugging Face NewsCLIPpings dataset.
    
    Falls back gracefully to benchmark samples if offline or dataset requires VisualNews archive.
    """
    samples: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
        print(f"[*] Attempting to stream {max_samples} pairs for '{split}' split from Hugging Face ({dataset_name})...")
        hf_dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        
        count = 0
        for item in hf_dataset:
            # Check for image and caption keys in HF record
            image = item.get("image") or item.get("image_path")
            caption = item.get("caption") or item.get("text") or item.get("id")
            falsified = item.get("falsified", 0)

            if image is not None and caption is not None:
                samples.append({
                    "image": image,
                    "caption": caption,
                    "falsified": int(falsified)
                })
                count += 1
                if count >= max_samples:
                    break
        print(f"[+] Successfully loaded {len(samples)} streaming samples from {dataset_name}.")
    except Exception as e:
        print(f"[!] Hugging Face streaming warning: {e}")
        if fallback_benchmark:
            print(f"[*] Generating {max_samples} balanced benchmark samples for offline/local environment...")
            seed = 42 if split == "train" else (101 if split == "val" else 202)
            samples = generate_benchmark_samples(num_samples=max_samples, seed=seed)
        else:
            raise e

    return samples


def get_dataloaders(
    train_samples: Optional[List[Dict[str, Any]]] = None,
    val_samples: Optional[List[Dict[str, Any]]] = None,
    test_samples: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    model_name: str = "openai/clip-vit-base-patch32"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Constructs train, validation, and test PyTorch DataLoaders.
    
    If sample lists are omitted, balanced subsets (3500 train, 750 val, 750 test)
    are streamed / generated automatically.
    """
    preprocessor = CLIPDataPreprocessor(model_name=model_name)

    if train_samples is None:
        train_samples = stream_news_clippings_subset("train", max_samples=3500)
    if val_samples is None:
        val_samples = stream_news_clippings_subset("val", max_samples=750)
    if test_samples is None:
        test_samples = stream_news_clippings_subset("test", max_samples=750)

    train_dataset = NewsDataset(train_samples, preprocessor=preprocessor)
    val_dataset = NewsDataset(val_samples, preprocessor=preprocessor)
    test_dataset = NewsDataset(test_samples, preprocessor=preprocessor)

    # Use pin_memory if CUDA is available
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader, test_loader
