# Visual Object Search and Retrieval Using Local Feature Matching

**What it does:** Given a query image, search a UKBench image database and return the most visually related images (classical CV retrieval—no deep learning).

**Techniques:** SIFT local features, Bag-of-Visual-Words (2,000-word vocabulary), TF-IDF + cosine similarity, then **BFMatcher** KNN matching, Lowe’s ratio test, and RANSAC homography verification.

**Final architecture:** Two stages—(1) TF-IDF over all database vectors → top **574** candidates; (2) per-candidate SIFT matching and geometric inliers → rank by inlier count → Top-K. Implemented in `evaluate_100.py` and `retrieve_tfidf_ransac.py`.

**Main measured result:** Mean **Recall@10 = 0.78** on the 100-query protocol in `evaluate_100.py` (574 TF-IDF candidates, RANSAC inlier ranking). See [Results](#results) for details; this is not a full-dataset benchmark.

Course mini-project for **Foundations of Computer Vision (FCV)**.

---

## Key features

- Offline **SIFT** extraction over the full UKBench database (stored as a serialized feature cache).
- **Visual vocabulary** (2,000 words) built via **K-means** on a random sample of SIFT descriptors.
- Per-image **TF-IDF** vectors and **cosine similarity** for Stage 1 candidate selection.
- Optional **inverted index** construction and analysis (not used in the main evaluation retrieval loop).
- Stage 2 **BFMatcher KNN** matching, **Lowe’s ratio test** (0.75), and **RANSAC homography** (5 px threshold) with **inlier count** as the final rank score.
- **UKBench-style evaluation**: Recall@10 against the three other images in each 4-image relevance group.
- Experimental **HSV colour histogram** and **custom HOG** baselines (separate from the main SIFT pipeline).
- Earlier **brute-force SIFT** retrieval prototype (`src/experiments/retrieve.py`) for comparison and timing experiments.

---

## System architecture

### Final retrieval pipeline (current implementation)

Used by `evaluate_100.py`, `retrieve_tfidf_ransac.py`, and `evaluate_retrieval.py` for Stage 1–2 retrieval. Stage 2 descriptor matching uses **`cv2.BFMatcher`** KNN (`k=2`), not FLANN.

```
Query image
    ↓
SIFT
    ↓
Bag-of-Visual-Words / visual vocabulary (nearest word per descriptor)
    ↓
TF-IDF (L2-normalized query vector)
    ↓
Linear scan: cosine similarity vs all database TF-IDF vectors
    ↓
Top 574 candidates (default in eval scripts; configurable)
    ↓
BFMatcher KNN (k=2)
    ↓
Lowe ratio test (0.75)
    ↓
RANSAC + homography (5.0 px threshold)
    ↓
Geometric inlier count
    ↓
Final ranking (by inliers)
    ↓
Top-K results
```

The two-stage architecture reduces the number of expensive SIFT/RANSAC comparisons compared with exhaustive descriptor-level matching over the full database. Stage 1 still performs a **linear scan** over ~10,200 precomputed TF-IDF vectors (it does not query the inverted index).

### Development and earlier prototypes

**Brute-force SIFT prototype** (`src/experiments/retrieve.py`): Stage 1 compares the query to **every** database image using BFMatcher KNN + ratio test (match count = score), keeps the top **50** candidates, then RANSAC re-ranking. No TF-IDF or visual vocabulary.

```
Query → SIFT → [all images: BFMatcher KNN + ratio test → match count]
            → top 50 → RANSAC inliers → Top-K
```

**Other experiments** (not the final pipeline): colour histograms, custom HOG, inverted-index candidate analysis, and FLANN timing investigations—see [Development experiments](#development-experiments).

---

## Methodology

### SIFT

- OpenCV `cv2.SIFT_create()` on grayscale images.
- Each database entry stores `keypoints` as `(x, y)` float32 arrays and `descriptors` as SIFT descriptor matrices (`precompute_features.py`).
- Default OpenCV SIFT parameters (no custom tuning in code).

### Visual vocabulary / Bag-of-Visual-Words

1. `sample_sift_descriptors.py` draws **100,000** descriptors uniformly at random from the full SIFT cache (seed **42**).
2. `build_visual_vocabulary.py` runs OpenCV **K-means** with **K = 2000**, `KMEANS_PP_CENTERS`, up to **20** iterations, epsilon **0.1**, **3** restarts.
3. Each SIFT descriptor is assigned to its nearest visual word via `BFMatcher` with `NORM_L2` (`match`, one neighbour).

### TF-IDF

- **Term frequency**: for each image, count assignments per visual word; `tf = count / total_words_in_image`.
- **IDF**: `log((N + 1) / (df + 1)) + 1` where `N` is the number of images and `df` is document frequency per word (`build_tfidf_index.py`).
- **TF-IDF weight**: `tf * idf` per dimension; vectors are **L2-normalized** so retrieval uses **cosine similarity** (`np.dot` on normalized vectors).
- Query vectors are built the same way at retrieval time.

### Inverted index

- `src/experiments/build_inverted_index.py` maps each visual word id → list of image filenames that contain that word.
- `src/experiments/test_inverted_index.py` measures how many images share at least one query word (candidate reduction).
- **Main retrieval scripts** (`evaluate_100.py`, `retrieve_tfidf_ransac.py`, `evaluate_retrieval.py`) **do not query the inverted index**; Stage 1 still scans all precomputed TF-IDF vectors.

### Descriptor matching

- **Final pipeline (Stage 2):** `cv2.BFMatcher()` or `BFMatcher(NORM_L2)` with **KNN** (`k=2`) in `evaluate_100.py`, `retrieve_tfidf_ransac.py`, and related scripts.
- **Visual-word assignment (BoVW / query encoding):** also uses `BFMatcher` nearest-neighbour assignment to vocabulary centres.

> FLANN-based approximate nearest-neighbour matching was investigated during development to address the descriptor-matching bottleneck (e.g. KD-tree configuration in `src/experiments/retrieve.py`: `trees=5`, `checks=25`). **`get_good_matches` uses BFMatcher only**—FLANN is instantiated but not used on the current code path. The current retrieval implementation uses **BFMatcher** for candidate verification in the TF-IDF + RANSAC pipeline.

### Lowe’s ratio test

- A match is kept if `m.distance < 0.75 * n.distance` for the two nearest neighbours (consistent across retrieval scripts).

### RANSAC / homography

- `cv2.findHomography(..., cv2.RANSAC, 5.0)` when there are at least **4** good matches.
- **Score**: number of inliers (`sum(mask)`). If homography fails, score is **0**.

### Ranking

| Script / stage | Stage 1 score | Final rank score |
|----------------|---------------|------------------|
| `evaluate_100.py`, `retrieve_tfidf_ransac.py` | TF-IDF cosine | RANSAC inliers |
| `src/experiments/retrieve.py` | Good match count (all DB) | RANSAC inliers |
| `evaluate_retrieval.py` | TF-IDF (same as above) | Experiments: inliers, good matches, ratios, and normalized combinations |

---

## Results

| Metric | Configuration | Result |
|---|---|---:|
| Mean Recall@10 | 100 queries, 574 TF-IDF candidates | **0.78** |
| Final ranking | RANSAC homography inliers | — |
| Database | UKBench | ~10,200 images |
| Visual vocabulary | K-means | 2,000 words |

> The reported Recall@10 is specific to the 100-query evaluation protocol implemented in `evaluate_100.py`; it is not a full-dataset benchmark.

Mean Recall@10 **0.78** comes from aggregating per-query recall after a local run (`evaluate_100.py` → `results/evaluation_100_574.pkl`). That artifact is excluded from Git; reproduce once indexes are built. Protocol: queries `ukbench00000`, `ukbench00004`, …, `ukbench00396`; Top-K = 10; recall = (relevant images in Top-10) / 3, where relevant means the other three images in the same UKBench group of four.

---

## Dataset

**UKBench** (~10,200 images) is the intended database. Images are named `ukbench#####.jpg`. The benchmark groups images in sets of **four**; for a query, the **three other images in the same group** are treated as relevant (standard UKBench object-retrieval setup in this code).

The dataset is **not** included in this repository (size and licensing). Place it at:

```text
dataset/ukbench/
```

(ignored by Git via `.gitignore`).

---

## Repository structure

```text
FCV/
├── .gitignore          # ignores dataset/, results/*.pkl, results/*.npy
├── requirements.txt
├── README.md
├── dataset/            # local only — UKBench images
├── results/            # local only — generated features and indexes
└── src/                # run commands from this directory
    ├── precompute_features.py
    ├── sample_sift_descriptors.py
    ├── build_visual_vocabulary.py
    ├── build_tfidf_index.py
    ├── retrieve_tfidf_ransac.py
    ├── evaluate_100.py
    ├── evaluate_retrieval.py
    └── experiments/    # earlier approaches, baselines, debug/viz (optional)
```

Large generated artifacts (not on GitHub) include:

| Artifact | Produced by | Role |
|----------|-------------|------|
| `results/sift_features.pkl` | `precompute_features.py` | Full-database SIFT keypoints + descriptors |
| `results/sift_sample.npy` | `sample_sift_descriptors.py` | 100k descriptor sample for K-means |
| `results/visual_vocabulary_2000.npy` | `build_visual_vocabulary.py` | 2000 × 128 visual word centres |
| `results/tfidf_index_2000.pkl` | `build_tfidf_index.py` | TF-IDF vectors, IDF, document frequencies |
| `results/inverted_index_2000.pkl` | `src/experiments/build_inverted_index.py` | Word → image postings (optional) |
| `results/evaluation_100_574.pkl` | `evaluate_100.py` | Per-query metrics for 100-query run |
| `results/color_histograms.pkl`, `hog_features.pkl` | `src/experiments/` histogram / HOG scripts | Baseline experiments |

On a full UKBench run, `results/sift_features.pkl` is approximately **6.6 GB** on disk (**10,200** images, **13,689,992** SIFT descriptors). UKBench, this file, and other generated `results/*.pkl` / `results/*.npy` artifacts are **not** in the repository—they are reproducible outputs of the scripts above and are excluded via `.gitignore`. Regenerating the SIFT cache requires significant time and disk space.

---

## Installation

Python **3** with dependencies listed in `requirements.txt`:

| Package | Role |
|---------|------|
| **opencv-python** (`cv2`) | SIFT, matching, K-means, homography |
| **numpy** | Arrays, TF-IDF vectors, descriptor data |
| **matplotlib** | Optional visualization scripts in `src/experiments/` (`sift_test.py`, `matching_test.py`, `inspect_dataset.py`) |

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use a recent `opencv-python` build that exposes `cv2.SIFT_create()` (on some platforms you may need `opencv-contrib-python` instead if SIFT is unavailable).

---

## Setup

1. Clone the repository.
2. Obtain UKBench and extract images to `dataset/ukbench/`.
3. Create an empty `results/` directory at the repo root (if missing):

   ```bash
   mkdir -p results
   ```

4. Run indexing scripts from `src/` (see below). Paths in scripts are relative to `src/` (`../dataset/ukbench`, `../results/...`).

---

## How to run

All commands assume the current working directory is **`src/`**:

```bash
cd src
```

### End-to-end indexing (main pipeline)

Run in order:

```bash
python precompute_features.py          # SIFT over all images → sift_features.pkl
python sample_sift_descriptors.py      # random 100k sample → sift_sample.npy
python build_visual_vocabulary.py      # K-means → visual_vocabulary_2000.npy
python build_tfidf_index.py            # TF-IDF index → tfidf_index_2000.pkl
```

Optional (inverted index for analysis, not required for the final pipeline):

```bash
python experiments/build_inverted_index.py         # → inverted_index_2000.pkl
```

### Single-query retrieval (TF-IDF + RANSAC)

```bash
python retrieve_tfidf_ransac.py
```

Defaults: query `../dataset/ukbench/ukbench00000.jpg`, **574** TF-IDF candidates, **Top-10** by inliers. Edit constants at the top of the file to change the query or `CANDIDATE_COUNT`.

### Single-query retrieval (brute-force SIFT prototype)

Earlier exhaustive-matching approach (not required for the TF-IDF pipeline):

```bash
python experiments/retrieve.py
```

Uses all images in Stage 1 (ratio-test match counts), top **50** candidates, then RANSAC. Prints Stage 1 / Stage 2 timings.

### TF-IDF only (no geometric stage)

```bash
python experiments/test_tfidf_retrieval.py
```

### Evaluation

**100-query evaluation** (one query per group among the first 100 groups):

```bash
python evaluate_100.py
```

Writes `../results/evaluation_100_574.pkl` and prints aggregate Recall@10 and timing.

**10-query scoring comparison** (multiple rank combinations):

```bash
python evaluate_retrieval.py
```

### Experiments (optional)

Scripts under `src/experiments/` are **not** required to run the final retrieval or 100-query evaluation. Paths remain relative to `src/` (`../dataset`, `../results`), so invoke them from `src/` as below.

```bash
python experiments/precompute_histograms.py && python experiments/test_histogram_retrieval.py
python experiments/precompute_hog.py && python experiments/test_hog_retrieval.py
python experiments/test_inverted_index.py
python experiments/analyze_vocabulary.py
python experiments/keypoints_data_distribution.py
```

Debug / visualization: `experiments/matching_test.py`, `experiments/geometric_verification.py`, `experiments/sift_test.py`, `experiments/compare_images.py`, `experiments/debug_ransac.py`, `experiments/inspect_dataset.py`.

There is **no single CLI entry point**; the pipeline is a sequence of scripts as above.

---

## Evaluation

### Ground truth

For query index `q`, group id is `q // 4`. Relevant images are the other three filenames `ukbench{4*group + i:05d}.jpg` for `i ∈ {0,1,2,3}`, excluding the query.

### Recall@10

For each query:

```text
recall = (# relevant images in top 10) / 3
```

Final ranking in `evaluate_100.py` uses **RANSAC inlier count** only (not TF-IDF score).

### 100-query protocol (`evaluate_100.py`)

| Setting | Value |
|---------|--------|
| Queries | `ukbench00000`, `ukbench00004`, …, `ukbench00396` (100 queries) |
| TF-IDF candidates | **574** |
| Top-K | **10** |
| Query TF-IDF | Batched nearest-centre assignment (`np.linalg.norm` + `argmin`), equivalent to nearest visual word |

See [Results](#results) for the main Recall@10 figure and protocol summary.

`evaluate_retrieval.py` reports **candidate recall** (fraction of the three relevant images present in the 574 TF-IDF candidates) and compares alternative Stage 2 scoring functions over **10** queries.

---

## Development experiments

These live under `src/experiments/` and are optional.

- **Brute-force SIFT prototype** (`experiments/retrieve.py`): full-database Stage 1 matching—useful for correctness checks; Stage 1 cost grows linearly with database size (development timings on the order of **minutes** for ~10k images have been observed; exact times depend on hardware and load).
- **FLANN:** investigated as an approximate NN alternative during development; not used in the final TF-IDF + RANSAC path (see Descriptor matching above).
- **Colour / HOG:** global HSV 2D histogram (32×32, correlation metric) and a **manual** 128×128 HOG-like descriptor (16×16 cells, 9 bins)—standalone baselines, not fused into the TF-IDF pipeline.
- **Inverted index:** built (`experiments/build_inverted_index.py`) and analyzed (`experiments/test_inverted_index.py`, `experiments/analyze_vocabulary.py`) for candidate-set reduction; **primary retrieval does not use it**—Stage 1 scans all TF-IDF vectors.

---

## Computational and scalability considerations

| Stage | Cost driver | Notes |
|-------|-------------|--------|
| SIFT precompute | One-time, CPU + disk | Large `sift_features.pkl` |
| TF-IDF index build | One pass over all descriptors | Dominated by word assignment |
| Stage 1 retrieval | O(N) dot products, N ≈ 10,200 | Sub-second to ~1 s per query in 100-query eval means |
| Stage 2 | 574 × (KNN on SIFT + RANSAC) | Dominates end-to-end time (~10 s mean per query in the same eval) |

The two-stage design exists because **pairwise SIFT matching over the full database** is prohibitively slow; BoVW/TF-IDF narrows the set before geometric verification. This is a **more scalable approach than exhaustive descriptor-level matching**, but Stage 1 is still an O(N) linear scan over N ≈ 10,200 TF-IDF vectors—not a fully indexed large-scale search over millions of images.

---

## Limitations

- **Classical features only**—SIFT and hand-crafted baselines; no learned image embeddings.
- Retrieval quality depends on a fixed **2,000-word** vocabulary trained on a **100k** descriptor subsample.
- **TF-IDF candidate retrieval** performs a **linear scan** over all precomputed vectors; the inverted index is **not** integrated into the primary retrieval path (`evaluate_100.py`, `retrieve_tfidf_ransac.py`).
- **SIFT feature storage** is large (~6.6 GB for `sift_features.pkl` on full UKBench).
- **Evaluation** uses a **100-query subset** (first image of each of the first 100 groups), not all 10,200 images—see [Results](#results).
- No spatial verification beyond homography inliers; no query expansion.
- Scripts use hard-coded paths and constants; there is no unified configuration layer.
- Some demo scripts assume a display (`cv2.imshow`) and are not headless-friendly.
- `retrieve_tfidf_ransac.py` uses OpenCV `KeyPoint` objects for query coordinates in RANSAC while the database stores numpy `(x,y)`—consistent with `evaluate_100.py`’s storage format, but mixed representations across files.

---

## Future improvements

- Query the **inverted index** (or ANN on TF-IDF) instead of scanning all images in Stage 1.
- Unify retrieval into one module with CLI arguments (query path, K, candidate count).
- Pin OpenCV/NumPy versions in `requirements.txt` for stricter reproducibility.
- Integrate optional FLANN for Stage 2 matching on large candidate sets.
- Full UKBench evaluation (all 10,200 queries) and comparison to published baselines.
- Package homography verification and TF-IDF query encoding shared across scripts to reduce duplication.

---

## Technologies and concepts

- **Local features**: SIFT detection and description  
- **Matching**: brute-force KNN, Lowe ratio test  
- **Geometry**: homography, RANSAC inliers  
- **Retrieval**: visual vocabulary, BoVW, TF-IDF, cosine similarity  
- **Data structures**: inverted index (built for experiments; primary retrieval uses TF-IDF linear scan)  
- **Baselines**: colour histograms, gradient-orientation histograms  
- **Stack**: Python, OpenCV, NumPy  

---

## Academic context

Course mini-project for **Foundations of Computer Vision**, exploring classical pipelines for **object-level image retrieval** on UKBench: from exhaustive local-feature matching to a two-stage **filter-and-verify** architecture (TF-IDF over visual words, then RANSAC geometric verification). The repository is intended as a **portfolio** artifact: source and documentation are public; UKBench and multi-gigabyte generated feature/index files remain local and reproducible from the scripts above.
