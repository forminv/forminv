"""
Run all 9 models on the FormInv 103-theorem dataset.
Saves results to artifacts/all_models_103_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/noel.thomas/chaos-logic-bench/extensions/forminv")

from scripts.run_eval_v3 import run_all_models

providers = [
    "openai",
    "openai",
    "openai",
    "anthropic",
    "anthropic",
    "deepseek",
    "deepseek",
    "google",
    "openrouter",
]
models = [
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "deepseek-chat",
    "deepseek-reasoner",
    "gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]

print("Starting FormInv 103-theorem evaluation for 9 models...", flush=True)

results = run_all_models("data/generated/forminv_v3_103.jsonl", providers, models, use_cache=True)

out = Path("artifacts/all_models_103_results.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {out}", flush=True)

# Print per-model summary
print("\nPer-model summary (N=103 theorems):", flush=True)
print(f"{'Model':<45} {'acc':>7} {'ig':>7} {'scr':>7} {'n':>6}", flush=True)
print("-" * 80, flush=True)
for model_key, stats in sorted(results.get("per_model", {}).items(), key=lambda x: x[1].get("scr", 0), reverse=True):
    scr = stats.get("scr", "N/A")
    acc = stats.get("mean_accuracy", "N/A")
    ig = stats.get("mean_ig", "N/A")
    n = stats.get("n_theorems", "N/A")
    try:
        print(f"{model_key:<45} acc={acc:.3f} ig={ig:.3f} scr={scr:.3f} n={n}", flush=True)
    except (TypeError, ValueError):
        print(f"{model_key:<45} acc={acc} ig={ig} scr={scr} n={n}", flush=True)

print("\nDone.", flush=True)
