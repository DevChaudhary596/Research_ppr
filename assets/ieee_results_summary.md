# CSC301 Seminar Manuscript — Experimental Summary Tables

## 1. System Footprint & Latency Benchmark

| Metric | Observed Value |
| :--- | :--- |
| Compute Device | `cpu` |
| Frozen CLIP Backbone Params | 151,671,297 |
| Trainable Cross-Attention Params | 2,462,465 |
| Triage Threshold ($S_c$) | 0.45 |
| Fast Neural Screening (Mean / Median / P95) | 69.47 ms / 68.39 ms / 73.86 ms |
| Escalated Verification Agent Latency | 0.70 ms (`offline_fallback`) |
| Search API Calls Saved (Benchmark Split) | 100.00% |

## 2. Classification Performance Summary

| Metric | Value |
| :--- | :--- |
| Accuracy | 0.8750 |
| Macro-F1 | 0.8730 |
| Macro Precision | 0.9000 |
| Macro Recall | 0.8750 |
| ROC-AUC | 1.0000 |
