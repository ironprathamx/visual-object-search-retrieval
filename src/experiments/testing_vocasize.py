import gzip
import pickle
import numpy as np


# ============================================================
# PATHS
# ============================================================

RESULT_2K = "../results/retrieval_raw_300.pkl.gz"
RESULT_4K = "../results/retrieval_raw_300_4k.pkl.gz"

CANDIDATE_COUNT = 574


# ============================================================
# LOAD
# ============================================================

def load_results(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


results_2k = load_results(RESULT_2K)
results_4k = load_results(RESULT_4K)

print("Loaded:")
print("2K queries:", len(results_2k))
print("4K queries:", len(results_4k))


# ============================================================
# RELEVANCE
# ============================================================

def get_group(filename):
    # ukbench00000.jpg -> group 0
    index = int(filename[7:12])
    return index // 4


def is_relevant(query_id, candidate_id):
    return get_group(query_id) == get_group(candidate_id)


# ============================================================
# METRIC CALCULATION
# ============================================================

def evaluate(results):

    candidate_recalls = []
    all_three = []

    ns_scores = []

    recall1 = []
    recall4 = []
    recall10 = []

    precision1 = []
    precision4 = []
    precision10 = []

    reciprocal_ranks = []

    hit1 = []
    hit4 = []
    hit10 = []

    for query_id, data in results.items():

        candidates = data["candidates"]

        relevant = [
            is_relevant(query_id, c["candidate_id"])
            for c in candidates
        ]

        # ----------------------------------------------------
        # Candidate Recall @ 574
        # ----------------------------------------------------

        relevant_count = sum(relevant)

        candidate_recalls.append(
            relevant_count / 3
        )

        all_three.append(
            relevant_count == 3
        )

        # ----------------------------------------------------
        # Final ranking
        #
        # Per-query min-max normalization
        # + logarithmic scoring
        # ----------------------------------------------------

        tfidf = np.array(
            [c["tfidf_score"] for c in candidates],
            dtype=np.float32
        )

        good = np.array(
            [c["good_matches"] for c in candidates],
            dtype=np.float32
        )

        inliers = np.array(
            [c["ransac_inliers"] for c in candidates],
            dtype=np.float32
        )

        def minmax(x):

            minimum = x.min()
            maximum = x.max()

            if maximum == minimum:
                return np.zeros_like(x)

            return (x - minimum) / (maximum - minimum)

        tfidf_norm = minmax(tfidf)
        good_norm = minmax(good)
        inlier_norm = minmax(inliers)

        final_scores = (
            0.25 * tfidf_norm
            + 0.70 * np.log1p(good_norm)
            + 0.05 * np.log1p(inlier_norm)
        )

        ranking = np.argsort(
            final_scores
        )[::-1]

        # ----------------------------------------------------
        # Ranked relevance
        # ----------------------------------------------------

        ranked_relevant = [
            relevant[i]
            for i in ranking
        ]

        # ----------------------------------------------------
        # Recall / Precision
        # ----------------------------------------------------

        top1 = ranked_relevant[:1]
        top4 = ranked_relevant[:4]
        top10 = ranked_relevant[:10]

        r1 = sum(top1) / 1
        r4 = sum(top4) / 3
        r10 = sum(top10) / 3

        p1 = sum(top1) / 1
        p4 = sum(top4) / 4
        p10 = sum(top10) / 10

        recall1.append(r1)
        recall4.append(r4)
        recall10.append(r10)

        precision1.append(p1)
        precision4.append(p4)
        precision10.append(p10)

        # ----------------------------------------------------
        # Hit rate
        # ----------------------------------------------------

        hit1.append(
            int(sum(top1) > 0)
        )

        hit4.append(
            int(sum(top4) > 0)
        )

        hit10.append(
            int(sum(top10) > 0)
        )

        # ----------------------------------------------------
        # MRR
        # ----------------------------------------------------

        rr = 0.0

        for rank, is_rel in enumerate(
            ranked_relevant,
            start=1
        ):

            if is_rel:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

        # ----------------------------------------------------
        # UKBench N-S
        #
        # Query itself is excluded, so:
        #
        # N-S = 1 + number of relevant images
        #       in top 4
        # ----------------------------------------------------

        ns = 1 + sum(top4)

        ns_scores.append(ns)

    return {
        "candidate_recall": np.mean(candidate_recalls),
        "all_three": np.mean(all_three),

        "ns": np.mean(ns_scores),
        "ns_percent": np.mean(ns_scores) / 4,

        "r1": np.mean(recall1),
        "r4": np.mean(recall4),
        "r10": np.mean(recall10),

        "p1": np.mean(precision1),
        "p4": np.mean(precision4),
        "p10": np.mean(precision10),

        "mrr": np.mean(reciprocal_ranks),

        "hit1": np.mean(hit1),
        "hit4": np.mean(hit4),
        "hit10": np.mean(hit10),
    }


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating 2K...")
metrics_2k = evaluate(results_2k)

print("Evaluating 4K...")
metrics_4k = evaluate(results_4k)


# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 90)
print("                 2K vs 4K VOCABULARY")
print("=" * 90)

print(
    f"{'Metric':<25}"
    f"{'K=2000':>15}"
    f"{'K=4000':>15}"
    f"{'Change':>15}"
)

print("-" * 90)


def row(name, key, percent=False):

    a = metrics_2k[key]
    b = metrics_4k[key]

    change = b - a

    if percent:
        print(
            f"{name:<25}"
            f"{a*100:>14.2f}%"
            f"{b*100:>14.2f}%"
            f"{change*100:>+14.2f} pp"
        )
    else:
        print(
            f"{name:<25}"
            f"{a:>15.4f}"
            f"{b:>15.4f}"
            f"{change:>+15.4f}"
        )


row("Candidate Recall @ 574", "candidate_recall", True)
row("All 3 in Candidates", "all_three", True)

print("-" * 90)

row("N-S", "ns")
row("N-S %", "ns_percent", True)

print("-" * 90)

row("Recall @ 1", "r1", True)
row("Recall @ 4", "r4", True)
row("Recall @ 10", "r10", True)

print("-" * 90)

row("Precision @ 1", "p1", True)
row("Precision @ 4", "p4", True)
row("Precision @ 10", "p10", True)

print("-" * 90)

row("MRR", "mrr", True)

print("-" * 90)

row("Hit @ 1", "hit1", True)
row("Hit @ 4", "hit4", True)
row("Hit @ 10", "hit10", True)

print("=" * 90)


# ============================================================
# VERDICT
# ============================================================

print("\nVERDICT")

if metrics_4k["candidate_recall"] > metrics_2k["candidate_recall"]:
    print("4K improves Candidate Recall@574.")
else:
    print("4K does NOT improve Candidate Recall@574.")

if metrics_4k["ns"] > metrics_2k["ns"]:
    print("4K improves final N-S.")
else:
    print("4K does NOT improve final N-S.")

if metrics_4k["mrr"] > metrics_2k["mrr"]:
    print("4K improves MRR.")
else:
    print("4K does NOT improve MRR.")