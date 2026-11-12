"""Day 5 Verification Test: End-to-End Pipeline, Latency Benchmark & IEEE Assets."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from src.pipeline import (
    MultimodalVerificationPipeline,
    detect_and_verify,
    export_ieee_summary,
)


def test_day5_pipeline():
    print("=" * 65)
    print(" DAY 5 TEST: END-TO-END INFERENCE, BENCHMARK & IEEE EXPORT")
    print("=" * 65)

    # 1. Initialize unified pipeline
    print("\n[1/4] Initializing MultimodalVerificationPipeline...")
    pipeline = MultimodalVerificationPipeline()
    print(f"  [OK] Device: {pipeline.device} | Checkpoint: {pipeline.checkpoint_loaded}")

    # 2. Test detect_and_verify on sample inputs
    print("\n[2/4] Running detect_and_verify on sample posts...")
    img_a = Image.new("RGB", (224, 224), color=(40, 100, 200))
    cap_a = "Crowds gather at London's Trafalgar Square during a climate change demonstration."
    res_a = detect_and_verify(img_a, cap_a)
    print(
        f"  [OK] Sample A -> Score: {res_a['consistency_score']} | "
        f"Verdict: {res_a['verdict']} | Escalated: {res_a['escalated']} | "
        f"Fast Latency: {res_a['fast_latency_ms']} ms"
    )
    assert 0.0 <= res_a["consistency_score"] <= 1.0

    # Force escalation check with high threshold to verify full two-tier return structure
    res_forced = pipeline.detect_and_verify(img_a, cap_a, threshold=1.1)
    print(
        f"  [OK] Escalation check -> Verdict: {res_forced['verdict']} | "
        f"Provider: {res_forced['search_provider']} | "
        f"Total Latency: {res_forced['total_latency_ms']} ms"
    )
    assert res_forced["escalated"] is True
    assert res_forced["verdict"] == "Out-of-Context"

    # 3. Benchmark fast vs. escalated latency
    print("\n[3/4] Running latency benchmark (12 samples)...")
    bench_stats = pipeline.benchmark_latency(num_samples=12, run_agent_sample=True)
    print(f"  [OK] Fast Mean Latency:   {bench_stats['fast_mean_ms']} ms")
    print(f"  [OK] Fast Median Latency: {bench_stats['fast_median_ms']} ms")
    print(f"  [OK] Fast P95 Latency:    {bench_stats['fast_p95_ms']} ms")
    print(f"  [OK] Escalated Agent:     {bench_stats['escalated_agent_ms']} ms ({bench_stats['escalated_provider']})")
    print(f"  [OK] API Calls Saved:     {bench_stats['api_calls_saved_pct']}%")

    # 4. Export IEEE manuscript summary
    print("\n[4/4] Exporting IEEE manuscript summary table...")
    eval_metrics = None
    report_path = os.path.join("assets", "classification_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    summary_path = export_ieee_summary(
        eval_metrics=eval_metrics,
        benchmark_stats=bench_stats,
        output_path="assets/ieee_results_summary.md",
    )
    assert os.path.exists(summary_path)
    print(f"  [OK] Summary saved at: {summary_path}")

    print("\n" + "=" * 65)
    print(" [PASSED] ALL DAY 5 END-TO-END PIPELINE TESTS SUCCESSFUL!")
    print("=" * 65)


if __name__ == "__main__":
    test_day5_pipeline()

