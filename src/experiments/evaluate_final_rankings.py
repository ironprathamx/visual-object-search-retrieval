import gzip
import pickle
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RAW_PATH = "../results/retrieval_raw_2550.pkl.gz"
OUTPUT_PATH = "../results/final_ranking_results_2550.pkl"

# Weight step for additional formula sweep
WEIGHT_STEP = 0.05


# ============================================================
# LOAD DATA
# ============================================================

with gzip.open(RAW_PATH, "rb") as f:
    raw_data = pickle.load(f)

print(f"Loaded {len(raw_data)} queries")


# ============================================================
# RELEVANCE
# ============================================================

def get_group_id(filename):

    number = int(
        filename.replace("ukbench", "").replace(".jpg", "")
    )

    return number // 4


def is_relevant(query_filename, candidate_filename):

    return (
        get_group_id(query_filename)
        == get_group_id(candidate_filename)
    )


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


# ============================================================
# FINAL STRATEGIES
# ============================================================

STRATEGIES = {

    "Good Matches": {
        "tfidf": 0.00,
        "good": 1.00,
        "inliers": 0.00,
        "ratio": 0.00
    },

    "10T 80G 10I": {
        "tfidf": 0.10,
        "good": 0.80,
        "inliers": 0.10,
        "ratio": 0.00
    },

    "15T 70G 15I": {
        "tfidf": 0.15,
        "good": 0.70,
        "inliers": 0.15,
        "ratio": 0.00
    },

    "20T 60G 20I": {
        "tfidf": 0.20,
        "good": 0.60,
        "inliers": 0.20,
        "ratio": 0.00
    },

    "25T 50G 25I": {
        "tfidf": 0.25,
        "good": 0.50,
        "inliers": 0.25,
        "ratio": 0.00
    },

    "30T 40G 30I": {
        "tfidf": 0.30,
        "good": 0.40,
        "inliers": 0.30,
        "ratio": 0.00
    },

    "35T 40G 25I": {
        "tfidf": 0.35,
        "good": 0.40,
        "inliers": 0.25,
        "ratio": 0.00
    },

    "25T 70G 5I": {
        "tfidf": 0.25,
        "good": 0.70,
        "inliers": 0.05,
        "ratio": 0.00
    }
}


# ============================================================
# ADDITIONAL WEIGHT SWEEP
# ============================================================
#
# Generate every T/G/I combination in 5% increments
# where:
#
#       TF-IDF + Good Matches + Inliers = 1
#
# Ratio is excluded because previous experiments showed
# that it was not competitive.
#
# This gives a systematic search around the existing formulas.
# ============================================================

print()
print("=" * 80)
print("GENERATING 5% WEIGHT SWEEP")
print("=" * 80)

generated_count = 0

for t in range(0, 101, 5):

    for g in range(0, 101 - t, 5):

        i = 100 - t - g

        name = f"{t}T {g}G {i}I"

        # Don't overwrite manually named strategies
        if name not in STRATEGIES:

            STRATEGIES[name] = {
                "tfidf": t / 100,
                "good": g / 100,
                "inliers": i / 100,
                "ratio": 0.00
            }

            generated_count += 1


print(f"Total formulas evaluated: {len(STRATEGIES)}")
print(f"Additional formulas generated: {generated_count}")


# ============================================================
# EVALUATE ONE STRATEGY
# ============================================================

def evaluate_strategy(weights):

    totals = {

        "r1": 0.0,
        "r4": 0.0,
        "r10": 0.0,

        "p1": 0.0,
        "p4": 0.0,
        "p10": 0.0,

        "mrr": 0.0,

        "h1": 0.0,
        "h4": 0.0,
        "h10": 0.0,

        "rel4": 0.0,
        "rel10": 0.0,

        "ns": 0.0
    }

    distribution = {

        "3/3": 0,
        "2/3": 0,
        "1/3": 0,
        "0/3": 0
    }


    # --------------------------------------------------------
    # PROCESS EVERY QUERY
    # --------------------------------------------------------

    for query_filename, query_data in raw_data.items():

        candidates = query_data["candidates"]


        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        tfidf = np.array([
            c["tfidf_score"]
            for c in candidates
        ])

        good = np.array([
            c["good_matches"]
            for c in candidates
        ])

        inliers = np.array([
            c["ransac_inliers"]
            for c in candidates
        ])

        ratio = np.array([
            c["inlier_ratio"]
            for c in candidates
        ])


        # ----------------------------------------------------
        # NORMALIZE INDEPENDENTLY FOR THIS QUERY
        # ----------------------------------------------------

        tfidf = minmax(tfidf)
        good = minmax(good)
        inliers = minmax(inliers)
        ratio = minmax(ratio)


        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        scores = (

            weights["tfidf"] * tfidf

            + weights["good"] * good

            + weights["inliers"] * inliers

            + weights["ratio"] * ratio
        )


        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        order = np.argsort(-scores)

        ranked = [
            candidates[i]
            for i in order
        ]


        # ----------------------------------------------------
        # RELEVANCE
        # ----------------------------------------------------

        relevant = [

            is_relevant(
                query_filename,
                c["candidate_id"]
            )

            for c in ranked
        ]


        # ----------------------------------------------------
        # N-S SCORE
        # ----------------------------------------------------
        #
        # Your definition:
        #
        # N-S = number of relevant images in top 4
        #
        # Therefore:
        #
        # 0 <= N-S <= 3
        #
        # The query itself is not in the candidate list,
        # so only the other 3 group images can be retrieved.
        #
        # For percentage reporting, /4 is used because the
        # UKBench group contains 4 images in total.
        # ----------------------------------------------------

        ns = sum(relevant[:4])

        totals["ns"] += ns


        # ----------------------------------------------------
        # RECALL
        # ----------------------------------------------------

        r1 = sum(relevant[:1]) / 3
        r4 = sum(relevant[:4]) / 3
        r10 = sum(relevant[:10]) / 3


        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        p1 = sum(relevant[:1]) / 1
        p4 = sum(relevant[:4]) / 4
        p10 = sum(relevant[:10]) / 10


        # ----------------------------------------------------
        # HIT
        # ----------------------------------------------------

        h1 = int(any(relevant[:1]))
        h4 = int(any(relevant[:4]))
        h10 = int(any(relevant[:10]))


        # ----------------------------------------------------
        # MRR
        # ----------------------------------------------------

        rr = 0.0

        for rank, rel in enumerate(relevant, start=1):

            if rel:

                rr = 1.0 / rank
                break


        # ----------------------------------------------------
        # RELEVANT IMAGES
        # ----------------------------------------------------

        rel4 = sum(relevant[:4])
        rel10 = sum(relevant[:10])


        # ----------------------------------------------------
        # TOP-10 DISTRIBUTION
        # ----------------------------------------------------

        if rel10 == 3:

            distribution["3/3"] += 1

        elif rel10 == 2:

            distribution["2/3"] += 1

        elif rel10 == 1:

            distribution["1/3"] += 1

        else:

            distribution["0/3"] += 1


        # ----------------------------------------------------
        # ACCUMULATE
        # ----------------------------------------------------

        totals["r1"] += r1
        totals["r4"] += r4
        totals["r10"] += r10

        totals["p1"] += p1
        totals["p4"] += p4
        totals["p10"] += p10

        totals["mrr"] += rr

        totals["h1"] += h1
        totals["h4"] += h4
        totals["h10"] += h10

        totals["rel4"] += rel4
        totals["rel10"] += rel10


    # ========================================================
    # AVERAGE
    # ========================================================

    n = len(raw_data)

    for key in totals:

        totals[key] /= n


    return totals, distribution


# ============================================================
# RUN ALL STRATEGIES
# ============================================================

all_results = {}

total = len(STRATEGIES)

print()
print("=" * 80)
print("EVALUATING FORMULAS")
print("=" * 80)

for count, (name, weights) in enumerate(STRATEGIES.items(), start=1):

    print(
        f"[{count}/{total}] "
        f"{name}"
    )

    metrics, distribution = evaluate_strategy(weights)

    all_results[name] = {

        "weights": weights,

        "metrics": metrics,

        "distribution": distribution
    }


# ============================================================
# MAIN TABLE
# ============================================================

print()
print("=" * 125)
print("FINAL UKBENCH EVALUATION — 2550 QUERIES")
print("=" * 125)

print(
    f"{'Strategy':<22}"
    f"{'R@1':>9}"
    f"{'R@4':>9}"
    f"{'R@10':>9}"
    f"{'P@1':>9}"
    f"{'P@4':>9}"
    f"{'P@10':>9}"
    f"{'MRR':>9}"
    f"{'H@1':>9}"
    f"{'H@4':>9}"
    f"{'H@10':>9}"
    f"{'N-S':>9}"
)

print("-" * 125)


for name, result in all_results.items():

    m = result["metrics"]

    print(

        f"{name:<22}"

        f"{m['r1']:>9.4f}"
        f"{m['r4']:>9.4f}"
        f"{m['r10']:>9.4f}"

        f"{m['p1']:>9.4f}"
        f"{m['p4']:>9.4f}"
        f"{m['p10']:>9.4f}"

        f"{m['mrr']:>9.4f}"

        f"{m['h1']:>9.4f}"
        f"{m['h4']:>9.4f}"
        f"{m['h10']:>9.4f}"

        f"{m['ns']:>9.4f}"
    )


# ============================================================
# N-S RANKING
# ============================================================
#
# This is the important new section.
#
# Every formula is sorted by average N-S.
# ============================================================

ns_results = []

for name, result in all_results.items():

    weights = result["weights"]
    metrics = result["metrics"]

    ns_results.append({

        "name": name,

        "tfidf": weights["tfidf"],
        "good": weights["good"],
        "inliers": weights["inliers"],

        "ns": metrics["ns"],

        "ns_percentage":
            metrics["ns"] / 4 * 100,

        "r1": metrics["r1"],
        "r4": metrics["r4"],
        "r10": metrics["r10"],

        "mrr": metrics["mrr"]
    })


ns_results.sort(
    key=lambda x: x["ns"],
    reverse=True
)


# ============================================================
# ALL FORMULAS — SORTED BY N-S
# ============================================================

print()
print("=" * 110)
print("ALL FORMULAS SORTED BY AVERAGE N-S")
print("=" * 110)

print(

    f"{'Rank':>5}"
    f"{'Formula':<20}"
    f"{'TF-IDF':>9}"
    f"{'Good':>9}"
    f"{'Inliers':>9}"
    f"{'Avg N-S':>12}"
    f"{'N-S %':>10}"
    f"{'R@1':>10}"
    f"{'R@4':>10}"
    f"{'R@10':>10}"
    f"{'MRR':>10}"
)

print("-" * 110)


for rank, result in enumerate(ns_results, start=1):

    print(

        f"{rank:>5}"

        f"{result['name']:<20}"

        f"{result['tfidf']:>9.2f}"
        f"{result['good']:>9.2f}"
        f"{result['inliers']:>9.2f}"

        f"{result['ns']:>12.4f}"

        f"{result['ns_percentage']:>10.2f}%"

        f"{result['r1']:>10.4f}"
        f"{result['r4']:>10.4f}"
        f"{result['r10']:>10.4f}"

        f"{result['mrr']:>10.4f}"
    )


# ============================================================
# BEST N-S FORMULA
# ============================================================

best_ns = ns_results[0]

print()
print("=" * 80)
print("BEST FORMULA BY AVERAGE N-S")
print("=" * 80)

print(
    f"Formula:       {best_ns['name']}"
)

print(
    f"TF-IDF:        {best_ns['tfidf']:.2f}"
)

print(
    f"Good Matches:  {best_ns['good']:.2f}"
)

print(
    f"Inliers:       {best_ns['inliers']:.2f}"
)

print(
    f"Average N-S:   {best_ns['ns']:.4f} / 4"
)

print(
    f"N-S %:         {best_ns['ns_percentage']:.2f}%"
)

print(
    f"Recall@1:      {best_ns['r1']:.4f}"
)

print(
    f"Recall@4:      {best_ns['r4']:.4f}"
)

print(
    f"Recall@10:     {best_ns['r10']:.4f}"
)

print(
    f"MRR:           {best_ns['mrr']:.4f}"
)


# ============================================================
# ADDITIONAL METRICS
# ============================================================

print()
print("=" * 125)
print("ADDITIONAL METRICS")
print("=" * 125)


for name, result in all_results.items():

    m = result["metrics"]
    d = result["distribution"]

    print()
    print(name)

    print(
        f"Mean relevant Top-4:  "
        f"{m['rel4']:.4f}"
    )

    print(
        f"Mean relevant Top-10: "
        f"{m['rel10']:.4f}"
    )

    print(
        f"Mean N-S score:       "
        f"{m['ns']:.4f} / 4"
    )

    print(
        f"Mean N-S percentage:  "
        f"{m['ns'] / 4 * 100:.2f}%"
    )

    print(

        f"Top-10: "

        f"3/3 = {d['3/3']} "
        f"({d['3/3'] / len(raw_data) * 100:.2f}%), "

        f"2/3 = {d['2/3']} "
        f"({d['2/3'] / len(raw_data) * 100:.2f}%), "

        f"1/3 = {d['1/3']} "
        f"({d['1/3'] / len(raw_data) * 100:.2f}%), "

        f"0/3 = {d['0/3']} "
        f"({d['0/3'] / len(raw_data) * 100:.2f}%)"
    )


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_PATH, "wb") as f:

    pickle.dump(
        all_results,
        f
    )


print()
print("Saved:")
print(OUTPUT_PATH)