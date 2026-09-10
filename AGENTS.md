# AGENTS.MD: CONTEXT & EXECUTION DIRECTIVE

## 1. System Mission & Project Identity
* **Project Title:** Multimodal Inconsistency & Fake News Detection Using Cross-Modal Contrastive Learning and Verification Agents [cite: 1]
* **Course Context:** B.Tech CSE Third-Year Seminar (CSC301), 5th Semester [cite: 1]
* **Target Environment:** Python 3.10+, PyTorch 2.x, Hugging Face Ecosystem, Google Colab (Free-tier NVIDIA T4 GPU: 16GB VRAM, 12GB System RAM)
* **Execution Window:** 5-Day Fast-Track Delivery

---

## 2. Objective & Motivation: The "Why"
Digital misinformation online has shifted from digitally manipulated/photoshopped images toward **out-of-context pairing** (cheapfakes) [cite: 1]. Adversaries pair an authentic, pristine image from a historical event with a fabricated or misattributed caption describing a contemporary crisis [cite: 1].

Conventional AI systems fail to combat this:
1. **Unimodal Image Detectors:** Report 0% synthetic artifacts because the image is genuinely real [cite: 1].
2. **Unimodal Text Detectors:** Process the caption as grammatically intact, syntactically sound English [cite: 1].
3. **Monolithic MLLMs (13B+):** Fail due to prohibitive memory requirements, slow inference latency, and hallucinated justifications when dealing with real-time news [cite: 1].

### Target Objective
Build a lightweight, high-speed **multimodal triage and verification system** [cite: 1]:
* A fine-grained **Cross-Attention Contrastive Neural Network** screens single image-text pairs in under 50 ms to compute a cross-modal consistency score [cite: 1].
* An **Autonomous Fact-Checking Search Agent** is invoked conditionally only when the consistency score drops below an anomaly threshold ($S_c < 0.45$), fetching real-world evidence to generate citation-backed explanations while preserving compute and API costs [cite: 1].

---

## 3. Literature Foundation & Critical Research Gaps

The architecture is built upon and directly resolves the research gaps of 4 base papers:

### Paper 1: NewsCLIPpings (Luo et al., EMNLP) [cite: 1]
* **Findings:** Established the standard out-of-context multimodal dataset (~71,000 samples) [cite: 1]. Showed that basic CLIP global cosine similarity handles random domain mismatches well, but fails when image and false text share broad thematic categories (e.g., protest in London vs. protest in Madrid) [cite: 1].
* **Limitations/Gap:** Relies entirely on coarse global average-pooled vectors ($1 \times 768$), missing subtle, token-to-patch contradictions; outputs an opaque number without explanations [cite: 1].
* **Our Scope of Improvement:** Replace global pooling with a **Multi-Head Cross-Attention Layer** that computes dense attention between individual text word tokens and localized $14 \times 14$ visual patches [cite: 1].

### Paper 2: COSMOS (Aneja et al., CVPR) [cite: 1]
* **Findings:** Utilized visual entity grounding via bounding boxes to match named entities between two competing captions for the same photograph [cite: 1].
* **Limitations/Gap:** **Dual-Caption Dependency.** COSMOS requires two conflicting captions for the exact same picture; it completely fails on single-caption social media posts [cite: 1]. Bounding box extraction also incurs high compute latency [cite: 1].
* **Our Scope of Improvement:** Discard multi-caption requirements; train our cross-attention network to evaluate the **internal, self-contained coherence of a single image-caption pair** [cite: 1].

### Paper 3: SNIFFER (Qi et al., ACL/CIKM) [cite: 1]
* **Findings:** Instruction-tuned a 13B Multimodal LLM to output both classification labels and natural-language explanations of multimodal deception [cite: 1].
* **Limitations/Gap:** Requires high-end multi-GPU clusters (NVIDIA A100s) inaccessible in undergraduate environments [cite: 1]. Relies on static parametric memory, causing hallucinations when evaluating breaking or novel events [cite: 1].
* **Our Scope of Improvement:** **Decoupled Architecture.** A lightweight, fast classifier (~2M trainable parameters) runs on free T4 GPUs [cite: 1]. Dynamic external RAG retrieval is triggered only when inconsistency is detected, preventing hallucinations [cite: 1].

### Paper 4: FactAgent (Li et al., IEEE / arXiv) [cite: 1]
* **Findings:** Used autonomous multi-agent LLM systems to break down news claims, run live web search queries, and verify facts dynamically [cite: 1].
* **Limitations/Gap:** Text-only; completely blind to visual content [cite: 1]. Running multi-step web searches on every incoming post incurs unacceptable latency (15–30s) and high search API costs [cite: 1].
* **Our Scope of Improvement:** **Multimodal Triage Filter.** The high-speed neural network filters out genuine content immediately [cite: 1]. The agent is triggered strictly for discordant pairs ($S_c < 0.45$), reducing search API overhead by over 65% [cite: 1].

---

## 4. Technical Architecture Specifications

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

* **Dataset:** NewsCLIPpings (`g-luo/news-clippings` on Hugging Face Hub) [cite: 1]
* **Subset Size:** 5,000 balanced pairs (3,500 train, 750 val, 750 test) [cite: 1]
* **Loss Function:** Binary Cross-Entropy with Logits (`torch.nn.BCEWithLogitsLoss`)
* **Optimization:** AdamW ($lr=1\times 10^{-4}$, weight decay=$1\times 10^{-2}$)

---

## 5. 5-Day Implementation & Delivery Roadmap

### Day 1: Data Pipeline & Environment Preparation
* Set up Colab/local virtual environment with `torch`, `transformers`, `datasets`, `tavily-python`, `langchain`.
* Stream a 5,000-pair balanced subset of `g-luo/news-clippings` directly from Hugging Face [cite: 1].
* Construct `NewsDataset(Dataset)` returning preprocessed visual tensors ($3 \times 224 \times 224$) and tokenized text IDs (max length 77).
* Validate PyTorch `DataLoader` (Batch size: 32) without RAM bottlenecks.

### Day 2: Architecture Construction & Cross-Attention Head
* Load pre-trained `openai/clip-vit-base-patch32` and freeze all visual and text backbone weights [cite: 1].
* Extract unpooled visual patch embeddings ($49 \times 768$) and sequence token embeddings ($77 \times 768$).
* Implement `CrossAttentionClassifier(nn.Module)`:
  * `nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)` [cite: 1].
  * Sequence average pooling followed by 2-layer MLP projection [cite: 1].
* Run forward pass test ensuring trainable parameters remain strictly $\sim$2.1M.

### Day 3: Model Training, Evaluation & Weights Checkpointing
* Train projection head for 5 epochs on Colab T4 GPU (target runtime: ~40 minutes).
* Track training loss, validation BCE loss, Accuracy, and Macro-F1.
* Save best checkpoint (`best_cross_attention_model.pt`) based on validation F1 score.
* Generate evaluation assets: Confusion Matrix, ROC-AUC curve, and Classification Report on the 750 test pairs.

### Day 4: Triage Threshold Gate & Agentic Verification Engine
* Implement triage decision logic:
  ```python
  if consistency_score >= 0.45:
      return {"status": "Aligned / Authentic", "latency_ms": t_fast}
  else:
      return trigger_verification_agent(caption, misaligned_tokens)
  ```
* Construct LangChain agent powered by Gemini API or local Ollama with Tavily Search / DuckDuckGo tool [cite: 1].
* Define verification prompt: synthesizes search results into a concise 2-sentence debunking summary with authoritative citations [cite: 1].
* Test fallback pipeline on sample out-of-context news posts to verify factual grounding [cite: 1].

### Day 5: End-to-End Demo & Presentation Assets
* Build unified inference function: `detect_and_verify(image_path, text_caption)`.
* Benchmark latency differences: Fast screening path ($< 50\text{ ms}$) vs. Escalated search path ($1.5 - 3.5\text{ s}$).
* Export final tables and figures for the CSC301 seminar manuscript (IEEE double-column format).

---

## 6. Antigravity Agent Directives & Guardrails
When orchestrating sub-agents, code generation, or task execution in this repository:
1. **Never unfreeze CLIP backbone encoders:** Ensure `param.requires_grad = False` for all base CLIP layers to fit memory constraints and train in $< 1$ hour on T4.
2. **Handle Hugging Face dataset streaming cleanly:** Use `load_dataset('g-luo/news-clippings')` in a generator/subset fashion to avoid downloading redundant multi-gigabyte image dumps.
3. **Deterministic Triage Threshold:** Keep the decision gate boundary at $0.45$ by default, but expose it as a tunable parameter in configuration.
4. **Fallback API Safety:** Ensure search agent code includes robust try/except blocks (fallback to DuckDuckGo if Tavily API quota runs out).
