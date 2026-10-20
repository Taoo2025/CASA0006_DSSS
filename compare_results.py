#!/usr/bin/env python3
"""
Compare Baseline vs Tourism-Enhanced Models
============================================
Analyzes the impact of tourism clustering features on model performance.

Run: python compare_results.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("COMPARISON: BASELINE vs TOURISM-ENHANCED MODELS")
print("=" * 80)

# Load results
baseline_dir = Path('results_baseline')
tourism_dir = Path('results_with_tourism')

if not baseline_dir.exists() or not tourism_dir.exists():
    print("ERROR: Missing result directories. Run both scripts first:")
    print("  python run_ml_baseline.py")
    print("  python run_ml_with_tourism.py")
    exit(1)

# Load evaluation metrics
baseline_eval = pd.read_csv(baseline_dir / 'baseline_evaluation.csv', index_col=0)
tourism_eval = pd.read_csv(tourism_dir / 'with_tourism_evaluation.csv', index_col=0)

# Load metadata
with open(baseline_dir / 'baseline_metadata.json') as f:
    baseline_meta = json.load(f)
    
with open(tourism_dir / 'with_tourism_metadata.json') as f:
    tourism_meta = json.load(f)

# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("1. DATASET COMPARISON")
print("=" * 80)

print(f"\nListings: {baseline_meta['listings_count']:,}")
print(f"POIs: {baseline_meta['pois_count']:,}")
print(f"Tourism clusters found: {tourism_meta['n_tourism_clusters']}")
print(f"DBSCAN parameters: eps={tourism_meta['dbscan_eps']}m, min_samples={tourism_meta['dbscan_min_samples']}")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("2. MODEL PERFORMANCE COMPARISON")
print("=" * 80)

comparison_metrics = {}

for model_name in baseline_eval.index:
    baseline_row = baseline_eval.loc[model_name]
    tourism_row = tourism_eval.loc[model_name]
    
    comparison_metrics[model_name] = {
        'baseline_f1': baseline_row['f1'],
        'tourism_f1': tourism_row['f1'],
        'f1_delta': tourism_row['f1'] - baseline_row['f1'],
        'f1_delta_pct': ((tourism_row['f1'] - baseline_row['f1']) / baseline_row['f1'] * 100) if baseline_row['f1'] > 0 else 0,
        'baseline_roc_auc': baseline_row['roc_auc'],
        'tourism_roc_auc': tourism_row['roc_auc'],
        'roc_auc_delta': tourism_row['roc_auc'] - baseline_row['roc_auc'],
    }

comparison_df = pd.DataFrame(comparison_metrics).T

print("\nF1-Score Comparison:")
print("-" * 60)
for model_name, metrics in comparison_metrics.items():
    print(f"\n{model_name.upper()}")
    print(f"  Baseline F1:   {metrics['baseline_f1']:.4f}")
    print(f"  Tourism F1:    {metrics['tourism_f1']:.4f}")
    print(f"  Improvement:   {metrics['f1_delta']:+.4f} ({metrics['f1_delta_pct']:+.2f}%)")
    print(f"  ROC-AUC Δ:     {metrics['roc_auc_delta']:+.4f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("3. FEATURES COMPARISON")
print("=" * 80)

print(f"\nBaseline Features ({len(baseline_meta['features_numeric'])} numeric):")
for feat in baseline_meta['features_numeric']:
    print(f"  • {feat}")

print(f"\nTourism-Enhanced Features (NEW):")
tourism_new = set(tourism_meta['features_numeric']) - set(baseline_meta['features_numeric'])
for feat in tourism_new:
    print(f"  ✓ {feat}")

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("4. KEY INSIGHTS & CONCLUSIONS")
print("=" * 80)

# Find best model
best_baseline_model = comparison_df['baseline_f1'].idxmax()
best_baseline_f1 = comparison_df.loc[best_baseline_model, 'baseline_f1']

best_tourism_model = comparison_df['tourism_f1'].idxmax()
best_tourism_f1 = comparison_df.loc[best_tourism_model, 'tourism_f1']

avg_improvement = comparison_df['f1_delta_pct'].mean()
max_improvement = comparison_df['f1_delta_pct'].max()
min_improvement = comparison_df['f1_delta_pct'].min()

print(f"\n📊 Overall Performance:")
print(f"  • Best baseline model: {best_baseline_model.upper()} (F1={best_baseline_f1:.4f})")
print(f"  • Best tourism model: {best_tourism_model.upper()} (F1={best_tourism_f1:.4f})")
print(f"  • Average F1 improvement: {avg_improvement:+.2f}%")
print(f"  • Range: {min_improvement:+.2f}% to {max_improvement:+.2f}%")

if avg_improvement > 0:
    print(f"\n✅ CONCLUSION: Tourism features IMPROVE model performance")
    print(f"   Tourism proximity & clustering enhance prediction of high-activity listings")
else:
    print(f"\n⚠️  CONCLUSION: Tourism features have MINIMAL impact ({avg_improvement:+.2f}%)")
    print(f"   Other factors (price, location, room type) may be more important")

# ============================================================================
# SAVE SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("5. SAVING COMPARISON REPORT")
print("=" * 80)

summary_dir = Path('results_comparison')
summary_dir.mkdir(exist_ok=True)

# Save comparison CSV
comparison_df.to_csv(summary_dir / 'comparison.csv')
print(f"✓ Saved: {summary_dir / 'comparison.csv'}")

# Save full comparison
comparison_report = {
    'baseline_metrics': baseline_eval.to_dict(),
    'tourism_metrics': tourism_eval.to_dict(),
    'improvement_metrics': comparison_metrics,
    'summary': {
        'best_baseline_model': best_baseline_model,
        'best_baseline_f1': float(best_baseline_f1),
        'best_tourism_model': best_tourism_model,
        'best_tourism_f1': float(best_tourism_f1),
        'average_improvement_pct': float(avg_improvement),
        'max_improvement_pct': float(max_improvement),
        'min_improvement_pct': float(min_improvement),
        'conclusion': 'Tourism features improve performance' if avg_improvement > 0 else 'Tourism features have minimal impact'
    }
}

with open(summary_dir / 'comparison_report.json', 'w') as f:
    json.dump(comparison_report, f, indent=2)
print(f"✓ Saved: {summary_dir / 'comparison_report.json'}")

print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {summary_dir.absolute()}")
print("\n📝 Next: Review results and write final report/markdown")
