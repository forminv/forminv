# FormInv tasks for lm-evaluation-harness

These YAMLs make FormInv runnable with
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Prerequisites

1. Upload the dataset to the Hugging Face Hub (see `scripts/build_hf_dataset.py`
   and `hf_release/README.md`). It lives at `forminv/forminv`.

## Usage

```bash
# a single paraphrase stratum
lm_eval --model hf --model_args pretrained=<model> \
        --tasks forminv_canonical \
        --include_path lm_eval_tasks/forminv

# the grouped task (canonical + surface, size-weighted accuracy)
lm_eval --model hf --model_args pretrained=<model> \
        --tasks forminv \
        --include_path lm_eval_tasks/forminv
```

## Tasks

| Task | HF config | What it measures |
|------|-----------|------------------|
| `forminv_canonical` | `canonical` | Accuracy on canonical statements only. |
| `forminv_surface`   | `surface`   | Accuracy on surface paraphrases. |
| `forminv_all`       | `all`       | Accuracy over every item. |
| `forminv` (group)   | --           | Size-weighted accuracy over canonical + surface. |

## Note on invariance metrics

lm-eval reports per-item **accuracy**. FormInv's headline metrics -- the
**Conditional Inconsistency Rate (CIR)**, **Semantic Consistency Rate (SCR)**, and
**Invariance Gap (IG)** -- require grouping paraphrases by `theorem_id` and comparing
verdicts within a theorem, which is outside lm-eval's per-item scoring model. For
those, use the native FormInv harness:

```bash
forminv eval --dataset data/generated/forminv_v3_50.jsonl --models all
forminv cir --multisample
```

A large gap between `forminv_canonical` and `forminv_surface` accuracy is itself a
coarse invariance signal; CIR/SCR sharpen it by conditioning on canonical-correct
items.
