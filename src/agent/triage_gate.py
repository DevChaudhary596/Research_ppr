"""Deterministic triage gate and misaligned-token extraction for multimodal screening."""

import re
from typing import Any, Dict, List, Optional
import torch

from src.agent.verifier import VerificationAgent


def extract_misaligned_tokens(
    caption: str,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_weights: Optional[torch.Tensor] = None,
    tokenizer: Optional[Any] = None,
    top_k: int = 5,
) -> List[str]:
    """Identifies caption tokens with the weakest cross-attention grounding to visual patches.

    Args:
        caption: Raw caption string.
        input_ids: Optional tensor of shape (77,) or (1, 77).
        attention_mask: Optional tensor of shape (77,) or (1, 77).
        attention_weights: Optional cross-attention map of shape (77, 49) or (1, 77, 49).
        tokenizer: CLIP tokenizer instance to decode token IDs.
        top_k: Maximum number of low-alignment tokens to return.

    Returns:
        List of candidate misaligned token strings.
    """
    if (
        attention_weights is not None
        and input_ids is not None
        and tokenizer is not None
    ):
        weights = attention_weights.detach().cpu()
        ids = input_ids.detach().cpu()
        if weights.dim() == 3:
            weights = weights[0]
        if ids.dim() == 2:
            ids = ids[0]

        mask = (
            attention_mask.detach().cpu()[0]
            if (attention_mask is not None and attention_mask.dim() == 2)
            else (attention_mask.detach().cpu() if attention_mask is not None else None)
        )

        # Peak visual patch attention per text token: shape (seq_len,)
        token_visual_peak = weights.max(dim=-1).values
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

        scored_tokens: List[tuple] = []
        for idx in range(ids.shape[0]):
            if mask is not None and int(mask[idx].item()) == 0:
                continue
            tok_id = int(ids[idx].item())
            if tok_id in special_ids:
                continue

            decoded = tokenizer.decode([tok_id], skip_special_tokens=True).strip()
            cleaned = re.sub(r"[^\w\-']", "", decoded)
            if len(cleaned) < 3:
                continue

            score = float(token_visual_peak[idx].item())
            scored_tokens.append((score, cleaned))

        # Sort ascending: lowest attention onto visual patches first
        scored_tokens.sort(key=lambda pair: pair[0])

        selected: List[str] = []
        seen = set()
        for _, tok in scored_tokens:
            low = tok.lower()
            if low not in seen:
                seen.add(low)
                selected.append(tok)
            if len(selected) >= top_k:
                return selected

    # Fallback heuristic if attention map/tokenizer are omitted
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9\-']{2,}\b", caption)
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "during",
        "after", "before", "over", "under", "amid", "near", "are", "was", "were",
    }
    candidates = [w for w in words if w.lower() not in stopwords]
    # Prefer proper nouns / capitalized entities first
    prioritized = sorted(candidates, key=lambda w: (0 if w[0].isupper() else 1, -len(w)))
    deduped: List[str] = []
    seen = set()
    for w in prioritized:
        if w.lower() not in seen:
            seen.add(w.lower())
            deduped.append(w)
        if len(deduped) >= top_k:
            break
    return deduped


class TriageGate:
    """Deterministic threshold gate routing aligned vs. inconsistent image-text pairs.

    Guardrail:
        - Default threshold is 0.45 (`S_c >= 0.45` passes immediately without API calls;
          `S_c < 0.45` escalates to the autonomous verification agent).
    """

    def __init__(
        self,
        threshold: float = 0.45,
        verifier: Optional[VerificationAgent] = None,
    ):
        self.threshold = float(threshold)
        self.verifier = verifier or VerificationAgent()

    def evaluate(
        self,
        consistency_score: float,
        caption: str,
        fast_latency_ms: float = 0.0,
        misaligned_tokens: Optional[List[str]] = None,
        verifier: Optional[VerificationAgent] = None,
    ) -> Dict[str, Any]:
        """Applies the deterministic triage threshold and returns unified decision metadata."""
        score = float(consistency_score)

        if score >= self.threshold:
            return {
                "status": "Aligned / Authentic",
                "verdict": "Authentic",
                "consistency_score": round(score, 4),
                "confidence_score": round(score, 4),
                "threshold": self.threshold,
                "escalated": False,
                "fast_latency_ms": round(float(fast_latency_ms), 2),
                "agent_latency_ms": 0.0,
                "total_latency_ms": round(float(fast_latency_ms), 2),
                "misaligned_tokens": [],
                "rationale": (
                    f"Cross-modal consistency score ({score:.2f} >= {self.threshold:.2f}) indicates "
                    "aligned image-text context; no external search triggered."
                ),
                "citations": [],
                "search_provider": None,
                "llm_provider_used": None,
                "search_query": "",
                "warnings": [],
            }

        active_verifier = verifier or self.verifier
        verification = active_verifier.verify(
            caption=caption,
            consistency_score=score,
            misaligned_tokens=misaligned_tokens,
        )
        agent_latency = float(verification.get("agent_latency_ms", 0.0))

        return {
            "status": "Discrepancy Detected / Out-of-Context",
            "verdict": verification.get("verdict", "Out-of-Context"),
            "consistency_score": round(score, 4),
            "confidence_score": verification.get("confidence_score", round(1.0 - score, 4)),
            "threshold": self.threshold,
            "escalated": True,
            "fast_latency_ms": round(float(fast_latency_ms), 2),
            "agent_latency_ms": round(agent_latency, 2),
            "total_latency_ms": round(float(fast_latency_ms) + agent_latency, 2),
            "misaligned_tokens": misaligned_tokens or [],
            "rationale": verification.get("rationale", ""),
            "citations": verification.get("citations", []),
            "evidence": verification.get("evidence", []),
            "search_provider": verification.get("search_provider"),
            "llm_provider_used": verification.get("llm_provider_used"),
            "search_query": verification.get("search_query", ""),
            "warnings": verification.get("warnings", []),
        }


def trigger_verification_agent(
    caption: str,
    misaligned_tokens: Optional[List[str]] = None,
    consistency_score: float = 0.0,
    verifier: Optional[VerificationAgent] = None,
) -> Dict[str, Any]:
    """Convenience wrapper matching the AGENTS.md Day 4 escalation call."""
    active_verifier = verifier or VerificationAgent()
    return active_verifier.verify(
        caption=caption,
        consistency_score=consistency_score,
        misaligned_tokens=misaligned_tokens,
    )

