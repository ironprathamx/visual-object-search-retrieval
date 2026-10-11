import gzip
import pickle


RAW_RESULTS_PATH = "../results/retrieval_raw_300_2k.pkl.gz"


# ============================================================
# LOAD RAW RETRIEVAL RESULTS
# ============================================================

print("Loading raw retrieval results...")

with gzip.open(RAW_RESULTS_PATH, "rb") as f:
    raw_results = pickle.load(f)

print(f"Queries: {len(raw_results)}")


# ============================================================
# RELEVANCE
# ============================================================

def is_relevant(query_filename, candidate_filename):

    query_id = int(
        query_filename.replace("ukbench", "").replace(".jpg", "")
    )

    candidate_id = int(
        candidate_filename.replace("ukbench", "").replace(".jpg", "")
    )

    return query_id // 4 == candidate_id // 4


# ============================================================
# METRICS
# ============================================================

TOP_K_VALUES = [1, 4, 10, 50, 100, 200, 300, 400, 500, 574, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000]

totals = {
    k: 0
    for k in TOP_K_VALUES
}

# Number of queries where ALL 3 other relevant images
# are present within the candidate set
all_three_count = {
    k: 0
    for k in TOP_K_VALUES
}

# Number of relevant images found in candidate set
total_relevant_found = {
    k: 0
    for k in TOP_K_VALUES
}

# Distribution at K=574
distribution_574 = {
    "3/3": 0,
    "2/3": 0,
    "1/3": 0,
    "0/3": 0
}


# ============================================================
# PROCESS QUERIES
# ============================================================

for query_filename, query_data in raw_results.items():

    candidates = query_data["candidates"]

    for k in TOP_K_VALUES:

        top_k = candidates[:k]

        relevant_count = sum(
            is_relevant(
                query_filename,
                candidate["candidate_id"]
            )
            for candidate in top_k
        )

        # ----------------------------------------------------
        # Recall
        #
        # There are 3 relevant images OTHER than the query.
        # ----------------------------------------------------

        totals[k] += relevant_count / 3.0

        total_relevant_found[k] += relevant_count

        if relevant_count == 3:
            all_three_count[k] += 1

    # --------------------------------------------------------
    # Distribution at 574
    # --------------------------------------------------------

    relevant_574 = sum(
        is_relevant(
            query_filename,
            candidate["candidate_id"]
        )
        for candidate in candidates[:574]
    )

    if relevant_574 == 3:
        distribution_574["3/3"] += 1
    elif relevant_574 == 2:
        distribution_574["2/3"] += 1
    elif relevant_574 == 1:
        distribution_574["1/3"] += 1
    else:
        distribution_574["0/3"] += 1


# ============================================================
# PRINT RESULTS
# ============================================================

num_queries = len(raw_results)

print()
print("=" * 90)
print("CANDIDATE RECALL ANALYSIS")
print("=" * 90)

print(
    f"{'K':>6} "
    f"{'Recall':>12} "
    f"{'Recall %':>12} "
    f"{'All 3':>12} "
    f"{'All 3 %':>12}"
)

print("-" * 90)

for k in TOP_K_VALUES:

    recall = totals[k] / num_queries

    all_three = all_three_count[k] / num_queries

    print(
        f"{k:6} "
        f"{recall:12.4f} "
        f"{recall * 100:11.2f}% "
        f"{all_three_count[k]:12} "
        f"{all_three * 100:11.2f}%"
    )


# ============================================================
# 574-CANDIDATE DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("RELEVANT IMAGE DISTRIBUTION WITHIN TOP-574")
print("=" * 70)

for key in ["3/3", "2/3", "1/3", "0/3"]:

    count = distribution_574[key]

    percentage = count / num_queries * 100

    print(
        f"{key}: "
        f"{count:4} / {num_queries} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# AVERAGE NUMBER OF RELEVANT IMAGES
# ============================================================

avg_relevant_574 = (
    total_relevant_found[574]
    / num_queries
)

print()
print("=" * 70)
print("TOP-574 SUMMARY")
print("=" * 70)

print(
    f"Average relevant images in top-574: "
    f"{avg_relevant_574:.4f} / 3"
)

print(
    f"Candidate Recall@574: "
    f"{avg_relevant_574 / 3 * 100:.2f}%"
)

print(
    f"Queries with all 3 relevant images: "
    f"{all_three_count[574]} / {num_queries}"
    f" ({all_three_count[574] / num_queries * 100:.2f}%)"
)