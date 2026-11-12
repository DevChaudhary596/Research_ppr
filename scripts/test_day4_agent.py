"""Day 4 Verification Test: Triage Threshold Gate and Autonomous Verification Agent."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.agent import (
    TriageGate,
    VerificationAgent,
    extract_misaligned_tokens,
    trigger_verification_agent,
)


def test_day4_agent():
    print("=" * 65)
    print(" DAY 4 TEST: TRIAGE THRESHOLD GATE & VERIFICATION AGENT")
    print("=" * 65)

    # 1. Test Misaligned Token Extraction (heuristic & attention-map path)
    print("\n[1/4] Testing misaligned token extraction...")
    sample_caption = "Protesters gather outside London Parliament during 2024 climate emergency."
    tokens_heuristic = extract_misaligned_tokens(caption=sample_caption, top_k=4)
    print(f"  [OK] Heuristic tokens: {tokens_heuristic}")
    assert len(tokens_heuristic) > 0

    dummy_ids = torch.tensor([[49406, 1200, 1500, 2100, 49407] + [0] * 72])
    dummy_mask = torch.tensor([[1, 1, 1, 1, 1] + [0] * 72])
    dummy_attn = torch.rand(1, 77, 49)

    class _MockTokenizer:
        all_special_ids = [49406, 49407, 0]

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            mapping = {1200: "London", 1500: "climate", 2100: "Parliament"}
            return mapping.get(ids[0], "")

    tokens_attn = extract_misaligned_tokens(
        caption=sample_caption,
        input_ids=dummy_ids,
        attention_mask=dummy_mask,
        attention_weights=dummy_attn,
        tokenizer=_MockTokenizer(),
        top_k=3,
    )
    print(f"  [OK] Attention-ranked misaligned tokens: {tokens_attn}")
    assert len(tokens_attn) == 3

    # 2. Test Triage Gate Aligned Fast Path (S_c >= 0.45)
    print("\n[2/4] Testing TriageGate Aligned Path (S_c >= 0.45)...")
    gate = TriageGate(threshold=0.45)
    aligned_res = gate.evaluate(
        consistency_score=0.82,
        caption=sample_caption,
        fast_latency_ms=18.4,
    )
    print(f"  [OK] Status: {aligned_res['status']} | Escalated: {aligned_res['escalated']}")
    assert aligned_res["escalated"] is False
    assert aligned_res["verdict"] == "Authentic"
    assert aligned_res["search_provider"] is None

    # 3. Test Tavily -> DuckDuckGo Fallback Safety (Guardrail 4)
    print("\n[3/4] Testing Tavily failure -> DuckDuckGo fallback safety...")
    agent = VerificationAgent(
        search_engine="tavily",
        fallback_search_engine="duckduckgo",
        max_search_results=2,
    )

    mock_ddg_results = [
        {
            "title": "Reuters Fact Check: Photo shows 2019 Madrid rally, not 2024 London",
            "url": "https://www.reuters.com/fact-check/example",
            "content": "The circulating photograph actually shows a 2019 demonstration in Madrid, not London.",
        }
    ]

    with patch.object(agent, "_search_tavily", side_effect=RuntimeError("Tavily quota exceeded")), \
         patch.object(agent, "_search_duckduckgo", return_value=mock_ddg_results):
        escalated_res = gate.evaluate(
            consistency_score=0.21,
            caption=sample_caption,
            fast_latency_ms=19.1,
            misaligned_tokens=tokens_attn,
            verifier=agent,
        )

    print(f"  [OK] Verdict: {escalated_res['verdict']} | Confidence: {escalated_res['confidence_score']}")
    print(f"  [OK] Fallback search provider: {escalated_res['search_provider']}")
    print(f"  [OK] Citations: {escalated_res['citations']}")
    print(f"  [OK] Rationale: {escalated_res['rationale']}")
    assert escalated_res["escalated"] is True
    assert escalated_res["verdict"] == "Out-of-Context"
    assert escalated_res["search_provider"] == "duckduckgo"
    assert len(escalated_res["citations"]) == 1

    # 4. Test Offline / Missing-Credentials Graceful Fallback
    print("\n[4/4] Testing offline graceful fallback (no API keys / no network)...")
    with patch.object(agent, "_search_tavily", side_effect=RuntimeError("No API key")), \
         patch.object(agent, "_search_duckduckgo", side_effect=RuntimeError("Offline")):
        offline_res = trigger_verification_agent(
            caption=sample_caption,
            misaligned_tokens=["London"],
            consistency_score=0.30,
            verifier=agent,
        )
    print(f"  [OK] Offline provider status: {offline_res['search_provider']}")
    print(f"  [OK] Warnings captured: {len(offline_res['warnings'])}")
    assert offline_res["search_provider"] == "offline_fallback"
    assert len(offline_res["warnings"]) == 2

    print("\n" + "=" * 65)
    print(" [PASSED] ALL DAY 4 TRIAGE & AGENT TESTS SUCCESSFUL!")
    print("=" * 65)


if __name__ == "__main__":
    test_day4_agent()

