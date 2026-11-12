"""End-to-end multimodal triage and verification pipeline (`detect_and_verify`)."""

import os
import statistics
import time
from typing import Any, Dict, List, Optional, Union

from PIL import Image
import torch

from src.agent.triage_gate import TriageGate, extract_misaligned_tokens
from src.agent.verifier import VerificationAgent
from src.data.dataset import generate_benchmark_samples
from src.data.preprocess import CLIPDataPreprocessor
from src.models.cross_attention import MultimodalInconsistencyDetector, count_parameters
from src.training.train import load_config


class MultimodalVerificationPipeline:
    """Unified two-tiered inference pipeline:

    Tier 1: Fast neural screening via Frozen CLIP ViT-B/32 + Cross-Attention Head.
    Tier 2: Conditional Verification Agent (Tavily -> DuckDuckGo + LangChain LLM)
            triggered strictly when S_c < triage_threshold (default 0.45).
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        checkpoint_path: Optional[str] = None,
        triage_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("data", {})
        train_cfg = self.config.get("training", {})
        agent_cfg = self.config.get("agent", {})

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        clip_model_name = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
        self.preprocessor = CLIPDataPreprocessor(
            model_name=clip_model_name,
            max_text_len=int(data_cfg.get("max_text_len", 77)),
            image_size=int(data_cfg.get("image_size", 224)),
        )

        self.model = MultimodalInconsistencyDetector(
            clip_model_name=clip_model_name,
            embed_dim=int(model_cfg.get("embed_dim", 768)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        self.checkpoint_loaded = self._load_checkpoint(checkpoint_path, train_cfg)
        self.model.to(self.device)
        self.model.eval()

        threshold = (
            float(triage_threshold)
            if triage_threshold is not None
            else float(agent_cfg.get("triage_threshold", 0.45))
        )

        self.verifier = VerificationAgent(
            search_engine=agent_cfg.get("search_engine", "tavily"),
            fallback_search_engine=agent_cfg.get("fallback_search_engine", "duckduckgo"),
            max_search_results=int(agent_cfg.get("max_search_results", 3)),
            llm_provider=agent_cfg.get("llm_provider", "gemini"),
            llm_model=agent_cfg.get("llm_model", "gemini-1.5-flash"),
        )
        self.triage_gate = TriageGate(threshold=threshold, verifier=self.verifier)

    def _load_checkpoint(
        self,
        explicit_path: Optional[str],
        train_cfg: Dict[str, Any],
    ) -> Optional[str]:
        """Loads trained classifier weights if a checkpoint exists on disk."""
        save_dir = train_cfg.get("save_dir", "checkpoints")
        default_ckpt = os.path.join(
            save_dir,
            train_cfg.get("checkpoint_filename", "best_cross_attention_model.pt"),
        )
        fallback_ckpt = os.path.join(save_dir, "test_day3_checkpoint.pt")

        candidates = [explicit_path] if explicit_path else [default_ckpt, fallback_ckpt]
        for path in candidates:
            if path and os.path.exists(path):
                payload = torch.load(path, map_location=self.device)
                if "classifier_state_dict" in payload:
                    self.model.classifier.load_state_dict(payload["classifier_state_dict"])
                elif "model_state_dict" in payload:
                    self.model.load_state_dict(payload["model_state_dict"], strict=False)
                print(f"[Pipeline] Loaded classifier checkpoint: {path}")
                return path

        print(
            "[Pipeline] Warning: No trained checkpoint found; using initialized cross-attention weights."
        )
        return None

    @torch.no_grad()
    def screen_pair(
        self,
        image: Union[str, Image.Image],
        text_caption: str,
        return_attention: bool = True,
    ) -> Dict[str, Any]:
        """Runs Tier-1 neural screening and measures forward-pass latency."""
        processed = self.preprocessor(image=image, text=text_caption)
        pixel_values = processed["pixel_values"].unsqueeze(0).to(self.device)
        input_ids = processed["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = processed["attention_mask"].unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        fast_latency_ms = (time.perf_counter() - t_start) * 1000.0

        score = float(outputs["consistency_score"].squeeze().item())
        attn_weights = outputs.get("attention_weights")

        tokenizer = getattr(self.preprocessor.processor, "tokenizer", None)
        misaligned_tokens = extract_misaligned_tokens(
            caption=text_caption,
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_weights=attn_weights,
            tokenizer=tokenizer,
            top_k=5,
        )

        return {
            "consistency_score": score,
            "fast_latency_ms": fast_latency_ms,
            "misaligned_tokens": misaligned_tokens,
        }

    def detect_and_verify(
        self,
        image_path: Union[str, Image.Image],
        text_caption: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Unified inference entry point: neural screening + conditional agent verification."""
        screen_out = self.screen_pair(
            image=image_path,
            text_caption=text_caption,
            return_attention=True,
        )

        if threshold is not None:
            gate = TriageGate(threshold=float(threshold), verifier=self.verifier)
        else:
            gate = self.triage_gate

        result = gate.evaluate(
            consistency_score=screen_out["consistency_score"],
            caption=text_caption,
            fast_latency_ms=screen_out["fast_latency_ms"],
            misaligned_tokens=screen_out["misaligned_tokens"],
        )
        result["device"] = str(self.device)
        result["checkpoint"] = self.checkpoint_loaded
        return result

    def benchmark_latency(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        num_samples: int = 16,
        run_agent_sample: bool = True,
    ) -> Dict[str, Any]:
        """Benchmarks Tier-1 neural screening latency vs. Tier-2 escalated verification."""
        bench_samples = samples or generate_benchmark_samples(num_samples=num_samples, seed=77)
        if not bench_samples:
            raise ValueError("Benchmark requires at least one sample.")

        # Warm-up pass (not recorded)
        _ = self.screen_pair(bench_samples[0]["image"], bench_samples[0]["caption"])

        fast_latencies: List[float] = []
        scores: List[float] = []
        escalated_count = 0

        for item in bench_samples:
            out = self.screen_pair(item["image"], item["caption"], return_attention=True)
            fast_latencies.append(out["fast_latency_ms"])
            scores.append(out["consistency_score"])
            if out["consistency_score"] < self.triage_gate.threshold:
                escalated_count += 1

        sorted_lat = sorted(fast_latencies)
        p95_idx = min(len(sorted_lat) - 1, max(0, int(0.95 * (len(sorted_lat) - 1))))

        escalated_latency_ms = 0.0
        escalated_provider = None
        if run_agent_sample:
            sample_item = bench_samples[min(1, len(bench_samples) - 1)]
            agent_out = self.verifier.verify(
                caption=sample_item["caption"],
                consistency_score=0.25,
                misaligned_tokens=["sample"],
            )
            escalated_latency_ms = float(agent_out["agent_latency_ms"])
            escalated_provider = agent_out["search_provider"]

        total_n = len(bench_samples)
        api_calls_saved_pct = ((total_n - escalated_count) / max(total_n, 1)) * 100.0
        trainable_p, frozen_p = count_parameters(self.model)

        return {
            "device": str(self.device),
            "num_samples": total_n,
            "triage_threshold": self.triage_gate.threshold,
            "fast_mean_ms": round(statistics.mean(fast_latencies), 2),
            "fast_median_ms": round(statistics.median(fast_latencies), 2),
            "fast_p95_ms": round(sorted_lat[p95_idx], 2),
            "escalated_agent_ms": round(escalated_latency_ms, 2),
            "escalated_provider": escalated_provider,
            "escalated_fraction": round(escalated_count / max(total_n, 1), 4),
            "api_calls_saved_pct": round(api_calls_saved_pct, 2),
            "trainable_params": trainable_p,
            "frozen_params": frozen_p,
        }


_DEFAULT_PIPELINE: Optional[MultimodalVerificationPipeline] = None


def detect_and_verify(
    image_path: Union[str, Image.Image],
    text_caption: str,
    checkpoint_path: Optional[str] = None,
    triage_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Top-level inference function specified in AGENTS.md Day 5."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None or checkpoint_path is not None:
        _DEFAULT_PIPELINE = MultimodalVerificationPipeline(
            checkpoint_path=checkpoint_path,
            triage_threshold=triage_threshold,
        )
    return _DEFAULT_PIPELINE.detect_and_verify(
        image_path=image_path,
        text_caption=text_caption,
        threshold=triage_threshold,
    )


def export_ieee_summary(
    eval_metrics: Optional[Dict[str, Any]] = None,
    benchmark_stats: Optional[Dict[str, Any]] = None,
    output_path: str = "assets/ieee_results_summary.md",
) -> str:
    """Exports Markdown/LaTeX-ready summary tables for the CSC301 seminar manuscript."""
    lines: List[str] = [
        "# CSC301 Seminar Manuscript — Experimental Summary Tables",
        "",
        "## 1. System Footprint & Latency Benchmark",
        "",
        "| Metric | Observed Value |",
        "| :--- | :--- |",
    ]

    if benchmark_stats:
        lines.extend(
            [
                f"| Compute Device | `{benchmark_stats.get('device', 'N/A')}` |",
                f"| Frozen CLIP Backbone Params | {benchmark_stats.get('frozen_params', 0):,} |",
                f"| Trainable Cross-Attention Params | {benchmark_stats.get('trainable_params', 0):,} |",
                f"| Triage Threshold ($S_c$) | {benchmark_stats.get('triage_threshold', 0.45)} |",
                f"| Fast Neural Screening (Mean / Median / P95) | "
                f"{benchmark_stats.get('fast_mean_ms', 0):.2f} ms / "
                f"{benchmark_stats.get('fast_median_ms', 0):.2f} ms / "
                f"{benchmark_stats.get('fast_p95_ms', 0):.2f} ms |",
                f"| Escalated Verification Agent Latency | {benchmark_stats.get('escalated_agent_ms', 0):.2f} ms "
                f"(`{benchmark_stats.get('escalated_provider', 'N/A')}`) |",
                f"| Search API Calls Saved (Benchmark Split) | {benchmark_stats.get('api_calls_saved_pct', 0):.2f}% |",
            ]
        )
    else:
        lines.append("| Status | Run `benchmark_latency()` to populate runtime statistics |")

    lines.extend(
        [
            "",
            "## 2. Classification Performance Summary",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
        ]
    )

    if eval_metrics:
        lines.extend(
            [
                f"| Accuracy | {eval_metrics.get('accuracy', 0.0):.4f} |",
                f"| Macro-F1 | {eval_metrics.get('macro_f1', 0.0):.4f} |",
                f"| Macro Precision | {eval_metrics.get('precision', 0.0):.4f} |",
                f"| Macro Recall | {eval_metrics.get('recall', 0.0):.4f} |",
                f"| ROC-AUC | {eval_metrics.get('roc_auc', 0.0):.4f} |",
            ]
        )
    else:
        lines.append("| Status | Run `evaluate_model()` to populate classification metrics |")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Pipeline] Exported IEEE manuscript summary: {output_path}")
    return output_path

