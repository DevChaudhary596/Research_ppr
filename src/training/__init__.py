"""Training and evaluation modules for Multimodal Inconsistency Detection."""

from src.training.train import train_model, train_one_epoch, validate, load_config
from src.training.evaluate import evaluate_model, run_inference_on_loader, plot_confusion_matrix, plot_roc_curve

__all__ = [
    "train_model",
    "train_one_epoch",
    "validate",
    "load_config",
    "evaluate_model",
    "run_inference_on_loader",
    "plot_confusion_matrix",
    "plot_roc_curve",
]
