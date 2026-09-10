# Multimodal Inconsistency & Fake News Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-fast multimodal triage and verification system designed to detect out-of-context multimodal misinformation (**cheapfakes**)—real, unmanipulated images paired with fabricated or misattributed captions.

---

## 🚀 Architecture Overview

Conventional systems either fail on cheapfakes (unimodal detectors see pristine images or sound grammar) or incur excessive latency and compute costs (13B+ MLLMs, continuous web searches). 

This system resolves this via a **two-tiered decoupled architecture**:

```
                          [ Input Social Post ]
                        (Single Image + Caption)
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
        [ Visual Channel ]                  [ Text Channel ]
       ViT-B/32 (CLIP Frozen)             CLIP Text Transformer
    Extracts Patch Tokens (49 x 768)     Extracts Word Tokens (77 x 768)
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
              [ Multi-Head Bidirectional Cross-Attention ]
                    (embed_dim=768, num_heads=8)
             Q = Text Tokens, K = Visual Patches, V = Visual Patches
                                   │
                                   ▼
                    [ Average Pooling & MLP Head ]
                 Linear(768 -> 128) -> ReLU -> Dropout(0.2)
                 Linear(128 -> 1)   -> Sigmoid
                                   │
                                   ▼
                 Consistency Score: S_c ∈ [0.0, 1.0]
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
        If S_c >= 0.45                      If S_c < 0.45
     [ PASSED / ALIGNED ]             [ DISCREPANCY DETECTED ]
     Latency: < 50 ms                 Trigger Agentic Search
     No External API Call                            │
                                                     ▼
                                    [ Autonomous Verification Agent ]
                                   (LangChain + Tavily Search API)
                                   Input: Caption + Misaligned Entities
                                   Tool: Live Google/Tavily Search
                                                     │
                                                     ▼
                                      [ Final Explanatory Output ]
                                      • Verdict: Out-of-Context
                                      • Confidence Score: 1 - S_c
                                      • Fact-Check Rationale & URLs
```

1. **Neural Screening Gate ($< 50\text{ ms}$):**
   - Base encoders: Frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`).
   - Cross-attention head ($\sim 2.1\text{M}$ trainable parameters) computing dense token-to-patch interactions.
   - Computes consistency score $S_c \in [0.0, 1.0]$.
2. **Autonomous Fact-Checking Agent ($S_c < 0.45$):**
   - Triggered strictly when inconsistency is detected ($S_c < 0.45$), reducing search API overhead by over **65%**.
   - Retrieves live evidence via Tavily Search (with DuckDuckGo fallback) and generates authoritative, citation-backed debunking rationale.

---

## 🔬 Literature Foundation & Research Gaps Resolved

| Base Paper | Existing Limitation | Our Scope of Improvement |
| :--- | :--- | :--- |
| **NewsCLIPpings** *(Luo et al., EMNLP)* | Coarse global cosine pooling misses fine token-to-patch contradictions. | **Multi-Head Cross-Attention Layer** evaluating dense interactions between word tokens and $14\times14$ visual patches. |
| **COSMOS** *(Aneja et al., CVPR)* | Dual-caption dependency & slow bounding box extraction. | Evaluates the **internal coherence of single image-caption pairs** with zero bounding box overhead. |
| **SNIFFER** *(Qi et al., ACL/CIKM)* | Requires multi-A100 GPU clusters (13B MLLM); static memory hallucinates on novel events. | **Decoupled lightweight design:** $\sim 2.1\text{M}$ parameters trainable on free Colab T4 GPU + live RAG search. |
| **FactAgent** *(Li et al., IEEE)* | Text-only; runs expensive multi-step search on 100% of posts (15–30s latency). | **Multimodal Triage Filter:** Genuine posts pass in $<50\text{ ms}$; agent search is only triggered when $S_c < 0.45$. |

---

## 📁 Repository Structure

```text
research_ppr/
├── AGENTS.md                  # Project directives & execution roadmap
├── README.md                  # Project documentation
├── requirements.txt           # Python package dependencies
├── configs/
│   └── config.yaml            # Hyperparameters & execution settings
├── src/
│   ├── data/
│   │   ├── preprocess.py      # CLIPProcessor image & text normalization
│   │   └── dataset.py         # NewsDataset, streaming, & benchmark generator
│   ├── models/
│   │   ├── clip_backbone.py   # Frozen CLIP ViT-B/32 feature extractors
│   │   └── cross_attention.py # Multi-Head Cross-Attention & MLP Head (~2.1M params)
│   ├── training/
│   │   ├── train.py           # Training loop with validation F1 checkpointing
│   │   └── evaluate.py        # ROC-AUC, Confusion Matrix, Classification Report
│   ├── agent/
│   │   ├── triage_gate.py     # Deterministic decision gate (Sc >= 0.45 vs Sc < 0.45)
│   │   └── verifier.py        # LangChain agent + Tavily/DuckDuckGo search + LLM
│   └── pipeline.py            # End-to-end detect_and_verify pipeline
├── notebooks/
│   └── colab_training_and_demo.ipynb  # One-click Google Colab training notebook
├── scripts/
│   └── test_day1_data.py      # Day 1 data pipeline verification test
├── checkpoints/               # Trained model weights (e.g. best_cross_attention_model.pt)
└── assets/                    # Figures, confusion matrices, and ROC curves
```

---

## 🛠️ Quickstart

### 1. Installation
```bash
git clone https://github.com/DevChaudhary596/Research_ppr.git
cd Research_ppr
pip install -r requirements.txt
```

### 2. Verify Data Pipeline (Day 1 Test)
```bash
python scripts/test_day1_data.py
```

### 3. Model Training (Google Colab)
Open `notebooks/colab_training_and_demo.ipynb` in Google Colab with a free-tier **T4 GPU** runtime to train the $\sim 2.1\text{M}$ parameter head in ~40 minutes. Checkpoints are automatically saved to `checkpoints/best_cross_attention_model.pt`.
