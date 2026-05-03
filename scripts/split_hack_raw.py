"""Split hack_raw bundle into train/val/test using only the hacking checkpoint's episodes.

Both honest and hacked episodes here come from the same policy weights, making
classification meaningful (not trivially separable by network identity).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hackrl.evaluation.activation_collection import load_bundle, save_bundle, split_bundle

INPUT = REPO_ROOT / "artifacts/activations/hack_raw/full_bundle.pt"
OUT_DIR = REPO_ROOT / "artifacts/activations/hack_only"

bundle = load_bundle(str(INPUT))
counts = bundle.label_counts()
print(f"Loaded bundle: {len(bundle.episodes)} episodes — {counts}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

train, val, test = split_bundle(bundle, seed=42)
save_bundle(train, str(OUT_DIR / "train.pt"))
save_bundle(val,   str(OUT_DIR / "val.pt"))
save_bundle(test,  str(OUT_DIR / "test.pt"))

print(f"Train: {len(train.episodes)}  {train.label_counts()}")
print(f"Val:   {len(val.episodes)}  {val.label_counts()}")
print(f"Test:  {len(test.episodes)}  {test.label_counts()}")
print(f"Saved to {OUT_DIR}/")
