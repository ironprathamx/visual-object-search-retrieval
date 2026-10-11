import gzip
import pickle
import numpy as np


RAW_PATH = "../results/retrieval_raw_300.pkl.gz"


# ============================================================
# LOAD RAW RETRIEVAL DATA
# ============================================================

with gzip.open(RAW_PATH, "rb") as f:
    raw_data = pickle.load(f)

print(type(raw_data))
print(f"Loaded {len(raw_data)} queries")

data = []

for query_filename, query_data in raw_data.items():
    query_data["query"] = query_filename
    data.append(query_data)

print(f"Prepared {len(data)} queries")

# ============================================================
# RELEVANCE
# ============================================================

def get_group_id(filename):
    number = int(filename.replace("ukbench", "").replace(".jpg", ""))
    return number // 4


def is_relevant(query_filename, candidate_filename):
    return get_group_id(query_filename) == get_group_id(candidate_filename)


# ============================================================
# NORMALIZATION
# ============================================================

def minmax(values):
    values = np.asarray(values, dtype=np.float64)

    vmin = values.min()
    vmax = values.max()

    if vmax == vmin:
        return np.zeros_like(values)

    return (values - vmin) / (vmax - vmin)


def rank_normalize(values):
    """
    Highest value -> 1
    Lowest value  -> 0
    """

    values = np.asarray(values, dtype=np.float64)

    order = np.argsort(-values)

    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))

    if len(values) == 1:
        return np.ones_like(values)

    return 1.0 - ranks / (len(values) - 1)


# ============================================================
# METRICS
# ============================================================

def evaluate_ranking(query_filename, ranked_candidates):

    relevant = [
        is_relevant(query_filename, c["candidate_id"])
        for c in ranked_candidates
    ]

    total_relevant = 3

    # ---------- Recall ----------
    r1 = sum(relevant[:1]) / total_relevant
    r4 = sum(relevant[:4]) / total_relevant
    r10 = sum(relevant[:10]) / total_relevant

    # ---------- Precision ----------
    p1 = sum(relevant[:1]) / 1
    p4 = sum(relevant[:4]) / 4
    p10 = sum(relevant[:10]) / 10

    # ---------- Hit ----------
    h1 = int(any(relevant[:1]))
    h4 = int(any(relevant[:4]))
    h10 = int(any(relevant[:10]))

    # ---------- MRR ----------
    rr = 0.0

    for rank, rel in enumerate(relevant, start=1):
        if rel:
            rr = 1.0 / rank
            break

    # ---------- Number of relevant images ----------
    rel4 = sum(relevant[:4])
    rel10 = sum(relevant[:10])

    # ---------- 3/3, 2/3, 1/3, 0/3 ----------
    top10_relevant = sum(relevant[:10])

    return {
        "r1": r1,
        "r4": r4,
        "r10": r10,

        "p1": p1,
        "p4": p4,
        "p10": p10,

        "mrr": rr,

        "h1": h1,
        "h4": h4,
        "h10": h10,

        "rel4": rel4,
        "rel10": rel10,

        "top10_relevant": top10_relevant
    }


# ============================================================
# WEIGHTED EVALUATION
# ============================================================

def evaluate_weights(weights, normalization="minmax"):

    w_tfidf, w_good, w_inliers, w_ratio = weights

    totals = {
        "r1": 0,
        "r4": 0,
        "r10": 0,

        "p1": 0,
        "p4": 0,
        "p10": 0,

        "mrr": 0,

        "h1": 0,
        "h4": 0,
        "h10": 0,

        "rel4": 0,
        "rel10": 0
    }

    for query in data:

        query_filename = query["query"]

        candidates = query["candidates"]

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

        ratio = np.array(
            [c["inlier_ratio"] for c in candidates],
            dtype=np.float64
        )

        if normalization == "minmax":
            tfidf_n = minmax(tfidf)
            good_n = minmax(good)
            inliers_n = minmax(inliers)
            ratio_n = minmax(ratio)

        elif normalization == "rank":
            tfidf_n = rank_normalize(tfidf)
            good_n = rank_normalize(good)
            inliers_n = rank_normalize(inliers)
            ratio_n = rank_normalize(ratio)

        score = (
            w_tfidf * tfidf_n
            + w_good * good_n
            + w_inliers * inliers_n
            + w_ratio * ratio_n
        )

        order = np.argsort(-score)

        ranked = [candidates[i] for i in order]

        metrics = evaluate_ranking(
            query_filename,
            ranked
        )

        for key in totals:
            totals[key] += metrics[key]

    n = len(data)

    for key in totals:
        totals[key] /= n

    return totals


# ============================================================
# GENERATE WEIGHTS
# ============================================================

def generate_weights(step=0.05):

    values = np.arange(
        0.0,
        1.0 + step / 2,
        step
    )

    weights = []

    for a in values:
        for b in values:
            for c in values:
                d = 1.0 - a - b - c

                if d < -1e-9:
                    continue

                if d > 1.0 + 1e-9:
                    continue

                weights.append((
                    round(float(a), 2),
                    round(float(b), 2),
                    round(float(c), 2),
                    round(float(d), 2)
                ))

    return weights


# ============================================================
# RUN SWEEP
# ============================================================

weights_list = generate_weights(step=0.05)

print(f"Testing {len(weights_list)} weight combinations...")


results_minmax = []
results_rank = []


for i, weights in enumerate(weights_list, start=1):

    metrics = evaluate_weights(
        weights,
        normalization="minmax"
    )

    results_minmax.append({
        "weights": weights,
        **metrics
    })

    metrics = evaluate_weights(
        weights,
        normalization="rank"
    )

    results_rank.append({
        "weights": weights,
        **metrics
    })

    if i % 100 == 0:
        print(f"Processed {i}/{len(weights_list)}")


# ============================================================
# PRINT TOP RESULTS
# ============================================================

def print_top(results, metric, title):

    sorted_results = sorted(
        results,
        key=lambda x: x[metric],
        reverse=True
    )

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"{'Weights (T,G,I,R)':<22}"
        f"{metric.upper():>10}"
        f"{'R@1':>10}"
        f"{'R@4':>10}"
        f"{'R@10':>10}"
        f"{'MRR':>10}"
        f"{'Rel@10':>10}"
    )

    print("-" * 90)

    for result in sorted_results[:10]:

        w = result["weights"]

        print(
            f"{str(w):<22}"
            f"{result[metric]:>10.4f}"
            f"{result['r1']:>10.4f}"
            f"{result['r4']:>10.4f}"
            f"{result['r10']:>10.4f}"
            f"{result['mrr']:>10.4f}"
            f"{result['rel10']:>10.4f}"
        )


# ============================================================
# MIN-MAX RESULTS
# ============================================================

print_top(
    results_minmax,
    "r1",
    "TOP WEIGHTS — MIN-MAX — R@1"
)

print_top(
    results_minmax,
    "r4",
    "TOP WEIGHTS — MIN-MAX — R@4"
)

print_top(
    results_minmax,
    "r10",
    "TOP WEIGHTS — MIN-MAX — R@10"
)

print_top(
    results_minmax,
    "mrr",
    "TOP WEIGHTS — MIN-MAX — MRR"
)


# ============================================================
# RANK-NORMALIZED RESULTS
# ============================================================

print_top(
    results_rank,
    "r1",
    "TOP WEIGHTS — RANK NORMALIZATION — R@1"
)

print_top(
    results_rank,
    "r4",
    "TOP WEIGHTS — RANK NORMALIZATION — R@4"
)

print_top(
    results_rank,
    "r10",
    "TOP WEIGHTS — RANK NORMALIZATION — R@10"
)

print_top(
    results_rank,
    "mrr",
    "TOP WEIGHTS — RANK NORMALIZATION — MRR"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open("../results/weight_sweep_300.pkl", "wb") as f:
    pickle.dump(
        {
            "minmax": results_minmax,
            "rank": results_rank
        },
        f
    )

print()
print("Saved:")
print("../results/weight_sweep_300.pkl")