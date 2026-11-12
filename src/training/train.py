"""Training pipeline for Multimodal Cross-Attention Classifier with checkpointing."""

import os
import sys
import time
from typing import Dict, Any, Optional, Tuple, List
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.cross_attention import MultimodalInconsistencyDetector
from src.data.dataset import NewsDataset, stream_news_clippings_subset, generate_benchmark_samples


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads configuration settings from YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Trains the trainable classifier head for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy, macro_f1).
    """
    model.train()
    # Keep backbone strictly in eval mode to prevent any BatchNorm/Dropout drift
    if hasattr(model, "backbone"):
        model.backbone.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in tqdm(dataloader, desc="Training Batch", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)  # Shape: (B, 1)

        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs["logits"]  # (B, 1)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item() * pixel_values.size(0)

        # Binary predictions: consistency score >= 0.5 (or logits >= 0.0)
        preds = (outputs["consistency_score"] >= 0.5).long().cpu().flatten().tolist()
        targets = labels.long().cpu().flatten().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets)

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / max(num_samples, 1)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, float(acc), float(f1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates the model on validation data.
    
    Returns:
        Dictionary containing loss, accuracy, macro_f1, precision, and recall.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs["logits"]
        loss = criterion(logits, labels)
        total_loss += loss.item() * pixel_values.size(0)

        preds = (outputs["consistency_score"] >= 0.5).long().cpu().flatten().tolist()
        targets = labels.long().cpu().flatten().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets)

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / max(num_samples, 1)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "val_loss": float(avg_loss),
        "val_acc": float(acc),
        "val_f1": float(f1),
        "val_precision": float(prec),
        "val_recall": float(rec),
    }


def train_model(
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "configs/config.yaml",
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    model: Optional[MultimodalInconsistencyDetector] = None,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Main training orchestration function with model checkpointing.
    
    Args:
        config: Configuration dictionary (loads from config_path if None).
        config_path: Path to YAML config file.
        train_loader: Optional DataLoader for training.
        val_loader: Optional DataLoader for validation.
        model: Optional instantiated MultimodalInconsistencyDetector.
        epochs: Number of epochs to train (overrides config if provided).
        device: Device string ('cuda', 'cpu', etc.).
        
    Returns:
        Dictionary containing training history and best validation metrics.
    """
    cfg = config or load_config(config_path)

    # 1. Device configuration
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    print(f"[Training] Using compute device: {target_device}")

    # 2. Hyperparameters
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    total_epochs = epochs or train_cfg.get("epochs", 5)
    learning_rate = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    save_dir = train_cfg.get("save_dir", "checkpoints")
    checkpoint_file = train_cfg.get("checkpoint_filename", "best_cross_attention_model.pt")
    checkpoint_path = os.path.join(save_dir, checkpoint_file)
    os.makedirs(save_dir, exist_ok=True)

    # 3. Initialize Model if not provided
    if model is None:
        print("[Training] Instantiating MultimodalInconsistencyDetector...")
        model = MultimodalInconsistencyDetector(
            clip_model_name=model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32"),
            embed_dim=model_cfg.get("embed_dim", 768),
            num_heads=model_cfg.get("num_heads", 8),
            mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 128),
            dropout=model_cfg.get("dropout", 0.2),
        )

    model.to(target_device)

    # Directive Guardrail Verification: Ensure all backbone weights remain frozen
    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
    print(f"[Training] Directive Guardrail - Backbone frozen: {backbone_frozen} | Classifier trainable: {classifier_trainable}")
    assert backbone_frozen, "CRITICAL ERROR: Backbone weights must be frozen to fit T4 memory constraints!"

    # 4. DataLoaders setup if not provided
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 0)

    if train_loader is None or val_loader is None:
        print("[Training] Loading dataset splits...")
        train_samples = stream_news_clippings_subset(
            split="train",
            max_samples=data_cfg.get("train_size", 3500),
            fallback_benchmark=True,
        )
        val_samples = stream_news_clippings_subset(
            split="val",
            max_samples=data_cfg.get("val_size", 750),
            fallback_benchmark=True,
        )
        train_dataset = NewsDataset(train_samples)
        val_dataset = NewsDataset(val_samples)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 5. Loss & Optimizer (AdamW + BCEWithLogitsLoss)
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # 6. Training Loop with Checkpointing on Validation Macro-F1
    print(f"\n[Training] Starting {total_epochs} epochs of training...")
    print(f"  - Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"  - Loss function: BCEWithLogitsLoss")
    print(f"  - Checkpoint destination: {checkpoint_path}\n")

    best_val_f1 = -1.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=target_device,
        )

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=target_device,
        )

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])
        history["val_f1"].append(val_metrics["val_f1"])

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] ({epoch_time:.1f}s) "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} "
            f"|| Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.4f} | Val F1: {val_metrics['val_f1']:.4f}"
        )

        # Save checkpoint if validation Macro-F1 improved
        if val_metrics["val_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1"]
            checkpoint_payload = {
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_acc"],
                "classifier_state_dict": model.classifier.state_dict(),
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }
            torch.save(checkpoint_payload, checkpoint_path)
            print(f"  --> Checkpoint saved to {checkpoint_path} (Best Val Macro-F1: {best_val_f1:.4f})")

    total_duration = time.time() - start_time
    print(f"\n[Training] Completed in {total_duration/60:.2f} minutes.")
    print(f"[Training] Best Validation Macro-F1 achieved: {best_val_f1:.4f}")

    return {
        "best_val_f1": best_val_f1,
        "checkpoint_path": checkpoint_path,
        "history": history,
        "total_duration_seconds": total_duration,
    }


if __name__ == "__main__":
    train_model()
