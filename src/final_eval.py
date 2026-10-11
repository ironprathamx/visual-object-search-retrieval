import json
import pickle
import time

from paths import DATASET_DIR, RESULTS_DIR, require_dir
from retrieve import VisualSearchEngine


# ============================================================
# CONFIGURATION
# ============================================================

NUM_IMAGES = 10200
IMAGES_PER_GROUP = 4 

# One query per UKBench group:
# 00000, 00004, 00008, ..., 10196
#
# This evaluates the first image of each group as the query.
# Therefore:
#   10200 / 4 = 2550 queries
QUERY_INDICES = range(0, NUM_IMAGES, IMAGES_PER_GROUP)

# Need at least 11 returned results to calculate @1, @4, @10.
TOP_K = 11


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_path(index):
    """
    Return the path to a UKBench image.
    """
    return DATASET_DIR / f"ukbench{index:05d}.jpg"


def get_relevant_images(query_index):
    """
    Return the 3 relevant DATABASE images for a query.

    UKBench has 4 images per group.

    Example for query 00000:

        00000  <- query
        00001  <- relevant
        00002  <- relevant
        00003  <- relevant

    These 3 images are therefore the relevant images used for
    Recall, Precision, Hit Rate, MRR, and mean relevant-image
    calculations.
    """

    group_start = (
        query_index // IMAGES_PER_GROUP
    ) * IMAGES_PER_GROUP

    relevant = set()

    for i in range(
        group_start,
        group_start + IMAGES_PER_GROUP
    ):
        if i != query_index:
            relevant.add(
                f"ukbench{i:05d}.jpg"
            )

    return relevant


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("FINAL FULL UKBENCH EVALUATION")
    print("=" * 70)

    print(f"Dataset images      : {NUM_IMAGES}")
    print(f"Queries evaluated   : {len(QUERY_INDICES)}")
    print(f"Images per group    : {IMAGES_PER_GROUP}")
    print("Relevant per query  : 3 other group images")
    print(f"Top-K               : {TOP_K}")

    print()

    print("Retrieval configuration:")
    print("  Vocabulary        : K = 4000")
    print("  Candidates        : 600")
    print("  Lowe ratio        : 0.80")
    print("  RANSAC threshold  : 5.0")

    print()

    print("=" * 70)

    require_dir(
        DATASET_DIR,
        "Download UKBench and extract images to dataset/ukbench/",
    )

    # --------------------------------------------------------
    # Initialize the ACTUAL retrieval engine (K=4000)
    # --------------------------------------------------------

    engine = VisualSearchEngine()

    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    num_queries = len(QUERY_INDICES)

    # Number of relevant images retrieved
    total_relevant_at_1 = 0
    total_relevant_at_4 = 0
    total_relevant_at_10 = 0

    # Number of queries with >= 1 relevant result
    hits_at_1 = 0
    hits_at_4 = 0
    hits_at_10 = 0

    # Mean Reciprocal Rank
    reciprocal_rank_sum = 0.0

    # Standard UKBench N-S score
    ns_sum = 0.0

    # Mean number of relevant images
    relevant_top4_sum = 0.0
    relevant_top10_sum = 0.0

    # Queries that fail
    failed_queries = []

    # Timing
    overall_start = time.time()

    # --------------------------------------------------------
    # PROCESS ALL QUERIES
    # --------------------------------------------------------

    for query_number, query_index in enumerate(
        QUERY_INDICES,
        start=1
    ):

        query_path = get_image_path(query_index)
        query_filename = query_path.name

        relevant = get_relevant_images(query_index)

        query_start = time.time()

        try:

            # =================================================
            # CALL THE ACTUAL RETRIEVAL ENGINE
            #
            # TOP_K = 10 allows us to calculate:
            #   Recall@1
            #   Recall@4
            #   Recall@10
            #   Precision@1
            #   Precision@4
            #   Precision@10
            #   Hit@1
            #   Hit@4
            #   Hit@10
            #   MRR
            #
            # retrieve.py already ranks all 600 candidates.
            # TOP_K only controls how many final results are
            # returned.
            # =================================================

            output = engine.search(
                str(query_path),
                TOP_K,
                INCLUDE_QUERY=True
            )

            results = output["results"]

            # ------------------------------------------------
            # Extract ranked filenames
            # ------------------------------------------------
            ranked_images = [
                result["image"]
                for result in results
            ]
            # =================================================
            # STANDARD UKBENCH N-S SCORE
            #
            # Standard UKBench:
            #
            #   4 images belong to each group.
            #   The query image itself is normally in the
            #   database and occupies one of the top-4 results.
            #
            #   N-S = number of correct group images in
            #         the first 4 returned images.
            #
            # Maximum N-S = 4.
            #
            # =================================================

            # Calculate N-S BEFORE removing query
            top4_for_ns = ranked_images[:4]

            ns_score = sum(
                image in relevant or image == query_filename
                for image in top4_for_ns
            )

            ns_sum += ns_score

            ranked_images = [
                image for image in ranked_images
                if image != query_filename
            ]

            # =================================================
            # TOP-1
            # =================================================

            top1 = ranked_images[:1]

            relevant_count_1 = sum(
                image in relevant
                for image in top1
            )

            total_relevant_at_1 += relevant_count_1

            if relevant_count_1 > 0:
                hits_at_1 += 1

            # =================================================
            # TOP-4
            #
            # Used for Recall@4, Precision@4, Hit@4,
            # and mean relevant images in Top-4.
            #
            # These metrics use the 3 OTHER relevant images
            # because the query itself is excluded from retrieval.
            # =================================================

            top4 = ranked_images[:4]

            relevant_count_4 = sum(
                image in relevant
                for image in top4
            )

            total_relevant_at_4 += relevant_count_4
            relevant_top4_sum += relevant_count_4

            if relevant_count_4 > 0:
                hits_at_4 += 1

            # =================================================
            # TOP-10
            # =================================================

            top10 = ranked_images[:10]

            relevant_count_10 = sum(
                image in relevant
                for image in top10
            )

            total_relevant_at_10 += relevant_count_10
            relevant_top10_sum += relevant_count_10

            if relevant_count_10 > 0:
                hits_at_10 += 1


            # =================================================
            # MRR
            #
            # Since the query itself is excluded, MRR measures
            # the rank of the first OTHER relevant image.
            # =================================================

            reciprocal_rank = 0.0

            for rank, image in enumerate(
                ranked_images,
                start=1
            ):

                if image in relevant:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_rank_sum += reciprocal_rank

            # =================================================
            # PROGRESS
            # =================================================

            query_time = time.time() - query_start

            if (
                query_number == 1
                or query_number % 25 == 0
                or query_number == num_queries
            ):

                elapsed = time.time() - overall_start

                average_time = (
                    elapsed / query_number
                )

                remaining = (
                    num_queries - query_number
                )

                eta = remaining * average_time

                print(
                    f"[{query_number:4d}/{num_queries}] "
                    f"{query_filename} | "
                    f"Top-4 relevant: "
                    f"{relevant_count_4}/3 | "
                    f"N-S: {ns_score}/4 | "
                    f"Time: {query_time:.2f}s | "
                    f"ETA: {eta / 60:.1f} min"
                )

        except Exception as e:

            failed_queries.append(
                (
                    query_index,
                    query_filename,
                    str(e)
                )
            )

            print(
                f"[ERROR] "
                f"{query_number}/{num_queries} "
                f"{query_filename}: {e}"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    successful_queries = (
        num_queries - len(failed_queries)
    )

    if successful_queries == 0:
        print("\nNo successful queries.")
        return

    # --------------------------------------------------------
    # Recall
    #
    # There are 3 relevant DATABASE images for each query
    # because the query itself is excluded from retrieval.
    #
    # Recall@K =
    # total relevant images retrieved at K
    # -----------------------------------
    # total relevant images available
    # --------------------------------------------------------

    recall_at_1 = (
        total_relevant_at_1
        / (3 * successful_queries)
    )

    recall_at_4 = (
        total_relevant_at_4
        / (3 * successful_queries)
    )

    recall_at_10 = (
        total_relevant_at_10
        / (3 * successful_queries)
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_at_1 = (
        total_relevant_at_1
        / successful_queries
    )

    precision_at_4 = (
        total_relevant_at_4
        / (4 * successful_queries)
    )

    precision_at_10 = (
        total_relevant_at_10
        / (10 * successful_queries)
    )

    # --------------------------------------------------------
    # Hit Rate
    # --------------------------------------------------------

    hit_at_1 = (
        hits_at_1
        / successful_queries
    )

    hit_at_4 = (
        hits_at_4
        / successful_queries
    )

    hit_at_10 = (
        hits_at_10
        / successful_queries
    )

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    mrr = (
        reciprocal_rank_sum
        / successful_queries
    )

    # --------------------------------------------------------
    # STANDARD UKBENCH N-S
    #
    # Maximum = 4.
    # --------------------------------------------------------

    ns_score = (
        ns_sum
        / successful_queries
    )

    ns_percentage = (
        ns_score / 4.0
    )

    # --------------------------------------------------------
    # Mean relevant images
    #
    # These are leave-one-out retrieval metrics and therefore
    # have a maximum of 3 relevant database images.
    # --------------------------------------------------------

    mean_relevant_top4 = (
        relevant_top4_sum
        / successful_queries
    )

    mean_relevant_top10 = (
        relevant_top10_sum
        / successful_queries
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    total_time = (
        time.time() - overall_start
    )

    print()
    print()
    print("=" * 70)
    print("FINAL FULL UKBENCH EVALUATION")
    print("=" * 70)

    print(
        f"Queries evaluated: "
        f"{successful_queries}"
    )

    print(
        f"Images in dataset: "
        f"{NUM_IMAGES}"
    )

    print(
        f"Groups evaluated: "
        f"{successful_queries}"
    )

    print(
        f"Failed queries: "
        f"{len(failed_queries)}"
    )

    print()

    # --------------------------------------------------------
    # N-S
    # --------------------------------------------------------

    print("--- UKBench N-S ---")

    print(
        f"Mean N-S score: "
        f"{ns_score:.4f} / 4"
    )

    print(
        f"Mean N-S percentage: "
        f"{ns_percentage * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # Retrieval performance
    # --------------------------------------------------------

    print("--- Retrieval Performance ---")

    print(
        f"Recall@1:  "
        f"{recall_at_1:.4f} "
        f"({recall_at_1 * 100:.2f}%)"
    )

    print(
        f"Recall@4:  "
        f"{recall_at_4:.4f} "
        f"({recall_at_4 * 100:.2f}%)"
    )

    print(
        f"Recall@10: "
        f"{recall_at_10:.4f} "
        f"({recall_at_10 * 100:.2f}%)"
    )

    print()

    print(
        f"Precision@1:  "
        f"{precision_at_1:.4f}"
    )

    print(
        f"Precision@4:  "
        f"{precision_at_4:.4f}"
    )

    print(
        f"Precision@10: "
        f"{precision_at_10:.4f}"
    )

    print()

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    print()

    print(
        f"Mean relevant images in Top-4: "
        f"{mean_relevant_top4:.4f} / 3"
    )

    print(
        f"Mean relevant images in Top-10: "
        f"{mean_relevant_top10:.4f} / 3"
    )

    print()

    # --------------------------------------------------------
    # Hit rates
    # --------------------------------------------------------

    print("--- Hit Rates ---")

    print(
        f"Hit@1:  "
        f"{hit_at_1 * 100:.2f}%"
    )

    print(
        f"Hit@4:  "
        f"{hit_at_4 * 100:.2f}%"
    )

    print(
        f"Hit@10: "
        f"{hit_at_10 * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    print("--- Runtime ---")

    print(
        f"Total evaluation time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Average time per query: "
        f"{total_time / successful_queries:.2f} seconds"
    )

    # --------------------------------------------------------
    # Failed queries
    # --------------------------------------------------------

    if failed_queries:

        print()
        print("--- Failed Queries ---")

        for (
            query_index,
            filename,
            error
        ) in failed_queries:

            print(
                f"{query_index}: "
                f"{filename} -> {error}"
            )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "configuration": {
            "dataset": "UKBench",
            "dataset_size": NUM_IMAGES,
            "queries": successful_queries,
            "vocabulary_size": 4000,
            "candidate_count": 600,
            "ratio_test": 0.80,
            "ransac_threshold": 5.0,
            "ranking": "0.25*tfidf + 0.70*log1p(good) + 0.05*log1p(inliers)",
        },
        "metrics": {
            "N-S": ns_score,
            "N-S_percentage": ns_percentage,
            "recall@1": recall_at_1,
            "recall@4": recall_at_4,
            "recall@10": recall_at_10,
            "precision@1": precision_at_1,
            "precision@4": precision_at_4,
            "precision@10": precision_at_10,
            "mrr": mrr,
            "mean_relevant@4": mean_relevant_top4,
            "mean_relevant@10": mean_relevant_top10,
            "hit@1": hit_at_1,
            "hit@4": hit_at_4,
            "hit@10": hit_at_10,
            "total_evaluation_time_seconds": total_time,
        },
        "failed_queries": [
            {"index": i, "filename": name, "error": err}
            for i, name, err in failed_queries
        ],
    }
    out_pkl = RESULTS_DIR / "final_eval_4k_summary.pkl"
    out_json = RESULTS_DIR / "final_eval_4k_summary.json"
    with open(out_pkl, "wb") as f:
        pickle.dump(summary, f)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print()
    print("Saved:")
    print(f"  {out_pkl}")
    print(f"  {out_json}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()