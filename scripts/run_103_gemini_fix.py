"""
Fix: Re-run Gemini on FormInv 103-theorem dataset with correct provider='gemini'.
Patches the google/gemini-2.5-flash entry in the saved results.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/noel.thomas/chaos-logic-bench/extensions/forminv")

from scripts.run_eval_v3 import run_all_models

print("Re-running gemini/gemini-2.5-flash with correct provider string...", flush=True)

results = run_all_models("data/generated/forminv_v3_103.jsonl", ["gemini"], ["gemini-2.5-flash"], use_cache=True)

# Load existing results and patch in the Gemini entry
out_path = Path("artifacts/all_models_103_results.json")
existing = json.loads(out_path.read_text())

# Remove the bad google/ entry and add the correct gemini/ entry
bad_key = "google/gemini-2.5-flash"
existing["per_model"].pop(bad_key, None)
existing["per_model"].update(results.get("per_model", {}))

out_path.write_text(json.dumps(existing, indent=2))
print(f"Patched and saved to {out_path}", flush=True)

# Print per-model summary
print("\nPer-model summary (N=103 theorems):", flush=True)
print(f"{'Model':<50} {'acc':>7} {'ig':>7} {'scr':>7} {'cov':>7} {'n':>6}", flush=True)
print("-" * 85, flush=True)
for model_key, stats in sorted(existing.get("per_model", {}).items(), key=lambda x: x[1].get("scr", 0), reverse=True):
    scr = stats.get("scr", "N/A")
    acc = stats.get("mean_accuracy", "N/A")
    ig = stats.get("mean_ig", "N/A")
    cov = stats.get("coverage", "N/A")
    n = stats.get("n_theorems", "N/A")
    try:
        print(
            f"{model_key:<50} acc={acc:.3f} ig={ig:.3f} scr={scr:.3f} cov={cov:.3f} n={n}",
            flush=True,
        )
    except (TypeError, ValueError):
        print(f"{model_key:<50} acc={acc} ig={ig} scr={scr} cov={cov} n={n}", flush=True)

print("\nDone.", flush=True)
