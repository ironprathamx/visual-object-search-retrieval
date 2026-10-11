import gzip
import pickle
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "../results/retrieval_raw_2550.pkl.gz"
OUTPUT_PATH = "../results/nonlinear_fine_sweep_2550.pkl"


# ============================================================
# LOAD RAW RESULTS
# ============================================================

with gzip.open(INPUT_PATH, "rb") as f:
    raw_results = pickle.load(f)

print(f"Loaded {len(raw_results)} queries")


# ============================================================
# HELPER: EXTRACT IMAGE NUMBER
# ============================================================

def get_image_number(filename):
    """
    Converts:
        ukbench00000.jpg
    into:
        0
    """
    return int(
        filename.replace("ukbench", "").replace(".jpg", "")
    )


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def min_max_normalize(values):
    values = np.asarray(values, dtype=np.float64)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


# ============================================================
# EVALUATE ONE RANKING
# ============================================================

def evaluate_ranking(ranked_candidates, query_id):

    # --------------------------------------------------------
    # Query image number
    # --------------------------------------------------------

    query_num = get_image_number(query_id)

    # UKBench:
    # 00000-00003 -> group 0
    # 00004-00007 -> group 1
    # etc.
    query_group = query_num // 4

    # --------------------------------------------------------
    # Three relevant images
    # --------------------------------------------------------

    relevant = {
        query_group * 4 + i
        for i in range(4)
        if query_group * 4 + i != query_num
    }

    # --------------------------------------------------------
    # Convert candidate filenames to image numbers
    # --------------------------------------------------------

    ranked_ids = [
        get_image_number(c["candidate_id"])
        for c in ranked_candidates
    ]

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    top1 = set(ranked_ids[:1])
    top4 = set(ranked_ids[:4])
    top10 = set(ranked_ids[:10])

    r1 = len(top1 & relevant) / 3
    r4 = len(top4 & relevant) / 3
    r10 = len(top10 & relevant) / 3

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    reciprocal_rank = 0.0

    for rank, candidate_id in enumerate(ranked_ids, start=1):

        if candidate_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    # --------------------------------------------------------
    # UKBench N-S
    #
    # N-S = 1 + number of relevant images in top 4
    # --------------------------------------------------------

    ns = 1 + len(top4 & relevant)

    return r1, r4, r10, reciprocal_rank, ns


# ============================================================
# PRECOMPUTE NORMALIZED + NONLINEAR SCORES
# ============================================================

prepared = {}

print("Preparing query scores...")

for query_id, result in raw_results.items():

    candidates = result["candidates"]

    # --------------------------------------------------------
    # Raw scores
    # --------------------------------------------------------

    tfidf = np.array(
        [c["tfidf_score"] for c in candidates],
        dtype=np.float64
    )

    good = np.array(
        [c["good_matches"] for c in candidates],
        dtype=np.float64
    )

    inliers = np.array(
        [c["ransac_inliers"] for c in candidates],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Per-query min-max normalization
    # --------------------------------------------------------

    tfidf_norm = min_max_normalize(tfidf)
    good_norm = min_max_normalize(good)
    inlier_norm = min_max_normalize(inliers)

    # --------------------------------------------------------
    # Nonlinear transformations
    #
    # Current best approach:
    #   log1p(Good Matches)
    #   log1p(RANSAC Inliers)
    # --------------------------------------------------------

    good_log = np.log1p(good_norm)
    inlier_log = np.log1p(inlier_norm)

    prepared[query_id] = {
        "candidates": candidates,
        "tfidf": tfidf_norm,
        "good_log": good_log,
        "inlier_log": inlier_log
    }


print("Score preparation complete.")


# ============================================================
# FINE SWEEP
# ============================================================

results = []

print("\nStarting fine sweep...")


# ------------------------------------------------------------
# Search region around current best:
#
# Current best:
#   TF-IDF = 25%
#   Good    = 70%
#   Inlier  = 5%
#
# Sweep:
#   TF-IDF = 22-28%
#   Good   = 65-75%
#   Inlier = remainder
# ------------------------------------------------------------

for tfidf_pct in range(22, 29):

    for good_pct in range(65, 76):

        inlier_pct = 100 - tfidf_pct - good_pct

        # Keep inlier weight within a reasonable range
        if inlier_pct < 0 or inlier_pct > 15:
            continue

        w_tfidf = tfidf_pct / 100.0
        w_good = good_pct / 100.0
        w_inlier = inlier_pct / 100.0

        total_r1 = 0.0
        total_r4 = 0.0
        total_r10 = 0.0
        total_mrr = 0.0
        total_ns = 0.0

        query_count = 0

        # ----------------------------------------------------
        # Evaluate every query
        # ----------------------------------------------------

        for query_id, data in prepared.items():

            candidates = data["candidates"]

            # -----------------------------------------------
            # Final nonlinear score
            # -----------------------------------------------

            final_scores = (
                w_tfidf * data["tfidf"]
                + w_good * data["good_log"]
                + w_inlier * data["inlier_log"]
            )

            # Higher score = better
            order = np.argsort(-final_scores)

            ranked_candidates = [
                candidates[i]
                for i in order
            ]

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            r1, r4, r10, mrr, ns = evaluate_ranking(
                ranked_candidates,
                query_id
            )

            total_r1 += r1
            total_r4 += r4
            total_r10 += r10
            total_mrr += mrr
            total_ns += ns

            query_count += 1

        # ----------------------------------------------------
        # Average metrics
        # ----------------------------------------------------

        avg_r1 = total_r1 / query_count
        avg_r4 = total_r4 / query_count
        avg_r10 = total_r10 / query_count
        avg_mrr = total_mrr / query_count
        avg_ns = total_ns / query_count

        results.append({
            "tfidf_weight": w_tfidf,
            "good_weight": w_good,
            "inlier_weight": w_inlier,

            "tfidf_pct": tfidf_pct,
            "good_pct": good_pct,
            "inlier_pct": inlier_pct,

            "ns": avg_ns,
            "ns_pct": avg_ns / 4 * 100,

            "recall@1": avg_r1,
            "recall@4": avg_r4,
            "recall@10": avg_r10,

            "mrr": avg_mrr
        })


# ============================================================
# SORT RESULTS BY N-S
# ============================================================

results.sort(
    key=lambda x: x["ns"],
    reverse=True
)


# ============================================================
# PRINT TOP 20
# ============================================================

print("\n")
print("=" * 100)
print("NONLINEAR FINE SWEEP — TOP 20 BY N-S")
print("=" * 100)

print(
    f"{'Formula':<18}"
    f"{'N-S':>9}"
    f"{'N-S %':>10}"
    f"{'R@1':>10}"
    f"{'R@4':>10}"
    f"{'R@10':>10}"
    f"{'MRR':>10}"
)

print("-" * 100)

for r in results[:20]:

    formula = (
        f"{r['tfidf_pct']}T "
        f"{r['good_pct']}G "
        f"{r['inlier_pct']}I"
    )

    print(
        f"{formula:<18}"
        f"{r['ns']:>9.4f}"
        f"{r['ns_pct']:>9.2f}%"
        f"{r['recall@1']:>10.4f}"
        f"{r['recall@4']:>10.4f}"
        f"{r['recall@10']:>10.4f}"
        f"{r['mrr']:>10.4f}"
    )


# ============================================================
# BEST RESULT BY EACH METRIC
# ============================================================

print("\n")
print("=" * 100)
print("BEST FORMULA BY EACH METRIC")
print("=" * 100)

metric_list = [
    ("N-S", "ns"),
    ("Recall@1", "recall@1"),
    ("Recall@4", "recall@4"),
    ("Recall@10", "recall@10"),
    ("MRR", "mrr")
]

for name, key in metric_list:

    best = max(
        results,
        key=lambda x: x[key]
    )

    print(
        f"{name:<12}: "
        f"{best['tfidf_pct']}T "
        f"{best['good_pct']}G "
        f"{best['inlier_pct']}I"
        f"  ->  {best[key]:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(results, f)

print("\n" + "=" * 100)
print(f"Saved results to: {OUTPUT_PATH}")
print("=" * 100)