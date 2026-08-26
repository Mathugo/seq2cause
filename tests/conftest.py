import sys
from pathlib import Path

# Run tests directly against `src/` without requiring the full (heavy)
# published dependency set (transformers, accelerate, datasets, captum) to be
# installed -- everything under test here only needs torch + jaxtyping.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
