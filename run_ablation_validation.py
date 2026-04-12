#!/usr/bin/env python3
"""
run_ablation_validation.py — Automated Ablation Validation Script

Runs walk-forward backtesting across 5 diverse tickers with 6 key
configurations to generate publication-ready results.

Usage:
    python run_ablation_validation.py

Output:
    results/ablation_full_results.csv
    results/ablation_averaged.csv
    results/ablation_table.tex
    results/ablation_significance.csv
"""

import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_runner import AblationRunner


def main():
    print("=" * 70)
    print("  ABLATION VALIDATION SUITE")
    print("  20 Tickers x 6 Configs -- Walk-Forward Backtesting (Large N)")
    print("=" * 70)

    # 20 diverse tickers spanning sectors & volatility regimes
    tickers = [
        # Tech (high growth, high vol)
        "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA",
        # Finance (moderate vol, rate-sensitive)
        "JPM", "GS", "BAC",
        # Healthcare (defensive, low vol)
        "JNJ", "PFE",
        # Energy (commodity-linked, cyclical)
        "XOM", "CVX",
        # Consumer (staples + discretionary)
        "WMT", "AMZN", "KO",
        # Industrials
        "CAT", "BA",
        # Commodities / ETFs
        "GLD", "SLV",
        # Crypto proxy
        "COIN",
    ]

    # 6 key configs: 3 baselines + 2 single models + 1 full ensemble
    configs = [
        "naive_baseline",       # Control: last close repeated
        "random_walk_baseline", # Control: random walk with drift
        "arima_baseline",       # Statistical baseline
        "mc_only",              # Monte Carlo standalone
        "ets_only",             # Exp Smoothing standalone
        "full_ensemble",        # Our system (ensemble + sentiment)
    ]

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTickers:  {', '.join(tickers)}")
    print(f"Configs:  {', '.join(configs)}")
    print(f"Output:   {output_dir}")
    print(f"Horizon:  21 days (1 month)")
    print(f"Step:     21 days")
    print(f"History:  3 years")
    print()

    # Initialize runner
    runner = AblationRunner(
        tickers=tickers,
        forecast_horizon=21,
        step_size=21,
        min_train_days=252,
        lookback_years=3,
    )

    # Progress callback for console
    start_time = time.time()

    def progress(pct, msg):
        elapsed = time.time() - start_time
        bar = "#" * int(pct * 30) + "-" * (30 - int(pct * 30))
        # Use ASCII-safe characters for Windows console
        safe_msg = msg.encode('ascii', 'replace').decode('ascii')
        print(f"\r  [{bar}] {pct*100:5.1f}%  {safe_msg:<50}", end="", flush=True)

    # Run suite
    print("Running ablation suite...\n")
    runner.run_suite(configs=configs, progress_callback=progress)

    print("\n\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Display averaged results
    avg = runner.get_averaged_comparison()
    if not avg.empty:
        print("\n--- Averaged Metrics (across all tickers) ---\n")
        print(avg.to_string(index=False))
    else:
        print("\n  [WARNING] No averaged results generated.")

    # Display full comparison
    full = runner.get_comparison_table()
    if not full.empty:
        print(f"\n--- Full Results: {len(full)} rows ---\n")
        print(full.to_string(index=False))

    # Significance tests
    sig_tests = runner.run_significance_tests(
        baseline_config="naive_baseline",
        test_config="full_ensemble")
    if sig_tests:
        print("\n--- Statistical Significance (Ensemble vs Naive) ---\n")
        for t in sig_tests:
            star = "***" if t['paired_ttest']['significant_5pct'] else "n.s."
            print(
                f"  {t['ticker']:6s}  "
                f"Naive MAE={t['baseline_mean_error']:.2f}  "
                f"Ensemble MAE={t['test_mean_error']:.2f}  "
                f"D={t['improvement_pct']:+.1f}%  "
                f"p={t['paired_ttest']['p_value']:.4f} {star}")

    # Export
    print(f"\n--- Exporting to {output_dir} ---\n")
    files = runner.export_results(output_dir)
    for name, path in files.items():
        print(f"  [OK] {name}: {path}")

    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  DONE in {elapsed_total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
