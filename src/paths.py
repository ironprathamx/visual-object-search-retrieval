"""Project-root paths for the visual retrieval engine.

All runtime artifacts live under <repo>/results and UKBench images
under <repo>/dataset/ukbench. Paths are resolved from this file so
scripts work regardless of the current working directory.
"""

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "ukbench"
RESULTS_DIR = PROJECT_ROOT / "results"

# Shipped (K = 4000) artifacts used by retrieve.py and the GUI
SIFT_FEATURES_PATH = RESULTS_DIR / "sift_features.pkl"
SIFT_SAMPLE_PATH = RESULTS_DIR / "sift_sample.npy"
VOCABULARY_PATH = RESULTS_DIR / "visual_vocabulary_4000.npy"
TFIDF_INDEX_PATH = RESULTS_DIR / "tfidf_index_4000.pkl"

# Historical K = 2000 artifacts (first full UKBench evaluation)
VOCABULARY_2000_PATH = RESULTS_DIR / "visual_vocabulary_2000.npy"
TFIDF_INDEX_2000_PATH = RESULTS_DIR / "tfidf_index_2000.pkl"


def require_file(path, how_to_build):
    """Return path if it exists; otherwise raise a setup-oriented error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n  {path}\n\n{how_to_build}"
        )
    return path


def require_dir(path, how_to_build):
    """Return directory path if it exists; otherwise raise a setup-oriented error."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory not found:\n  {path}\n\n{how_to_build}"
        )
    return path
