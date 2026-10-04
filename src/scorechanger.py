import pickle
import gzip
import os


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "../results/retrieval_raw_300.pkl.gz"


# ============================================================
# UKBENCH RELEVANCE
# ============================================================

def get_group_id(filename):

    # Example:
    #
    # ukbench00000.jpg -> group 0
    # ukbench00001.jpg -> group 0
    # ukbench00002.jpg -> group 0
    # ukbench00003.jpg -> group 0
    #
    # ukbench00004.jpg -> group 1
    # ...

    number = int(
        os.path.splitext(
            filename
        )[0].replace(
            "ukbench",
            ""
        )
    )

    return number // 4


def is_relevant(
    query_filename,
    candidate_filename
):

    return (
        get_group_id(query_filename)
        ==
        get_group_id(candidate_filename)
    )


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(
    candidates,
    key
):

    values = [
        float(c[key])
        for c in candidates
    ]

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:

        return [
            0.0
            for _ in values
        ]

    return [
        (x - minimum)
        /
        (maximum - minimum)
        for x in values
    ]


# ============================================================
# RANKING STRATEGIES
# ============================================================

def rank_direct(
    candidates,
    key
):

    return sorted(
        candidates,
        key=lambda c: c[key],
        reverse=True
    )


def rank_inliers_x_ratio(
    candidates
):

    return sorted(
        candidates,
        key=lambda c:
            c["ransac_inliers"]
            *
            c["inlier_ratio"],
        reverse=True
    )


def rank_weighted(
    candidates,
    tfidf_weight,
    ransac_weight,
    ratio_weight,
    good_weight
):

    # --------------------------------------------------------
    # Normalize each signal independently PER QUERY.
    # --------------------------------------------------------

    tfidf = minmax_normalize(
        candidates,
        "tfidf_score"
    )

    ransac = minmax_normalize(
        candidates,
        "ransac_inliers"
    )

    ratio = minmax_normalize(
        candidates,
        "inlier_ratio"
    )

    good = minmax_normalize(
        candidates,
        "good_matches"
    )

    scored = []

    for i, candidate in enumerate(
        candidates
    ):

        score = (

            tfidf_weight
            *
            tfidf[i]

            +

            ransac_weight
            *
            ransac[i]

            +

            ratio_weight
            *
            ratio[i]

            +

            good_weight
            *
            good[i]
        )

        scored.append(
            (
                candidate,
                score
            )
        )

    scored.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        candidate
        for candidate, score
        in scored
    ]


# ============================================================
# METRICS FOR ONE QUERY
# ============================================================

def evaluate_query(
    query_filename,
    ranked_candidates
):

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    top1 = ranked_candidates[:1]
    top4 = ranked_candidates[:4]
    top10 = ranked_candidates[:10]

    # --------------------------------------------------------
    # Relevant counts
    # --------------------------------------------------------

    relevant1 = sum(
        is_relevant(
            query_filename,
            c["candidate_id"]
        )
        for c in top1
    )

    relevant4 = sum(
        is_relevant(
            query_filename,
            c["candidate_id"]
        )
        for c in top4
    )

    relevant10 = sum(
        is_relevant(
            query_filename,
            c["candidate_id"]
        )
        for c in top10
    )

    # --------------------------------------------------------
    # UKBench has 3 relevant images OTHER than the query.
    # --------------------------------------------------------

    relevant1 = min(
        relevant1,
        3
    )

    relevant4 = min(
        relevant4,
        3
    )

    relevant10 = min(
        relevant10,
        3
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall1 = (
        relevant1 / 3.0
    )

    recall4 = (
        relevant4 / 3.0
    )

    recall10 = (
        relevant10 / 3.0
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision1 = (
        relevant1 / 1.0
    )

    precision4 = (
        relevant4 / 4.0
    )

    precision10 = (
        relevant10 / 10.0
    )

    # --------------------------------------------------------
    # Hit
    # --------------------------------------------------------

    hit1 = (
        1
        if relevant1 > 0
        else 0
    )

    hit4 = (
        1
        if relevant4 > 0
        else 0
    )

    hit10 = (
        1
        if relevant10 > 0
        else 0
    )

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    reciprocal_rank = 0.0

    for rank, candidate in enumerate(
        ranked_candidates,
        start=1
    ):

        if is_relevant(
            query_filename,
            candidate["candidate_id"]
        ):

            reciprocal_rank = (
                1.0 / rank
            )

            break

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {

        "recall@1":
            recall1,

        "recall@4":
            recall4,

        "recall@10":
            recall10,

        "precision@1":
            precision1,

        "precision@4":
            precision4,

        "precision@10":
            precision10,

        "mrr":
            reciprocal_rank,

        "hit@1":
            hit1,

        "hit@4":
            hit4,

        "hit@10":
            hit10,

        "relevant@4":
            relevant4,

        "relevant@10":
            relevant10
    }


# ============================================================
# EVALUATE ONE STRATEGY
# ============================================================

def evaluate_strategy(
    all_results,
    ranking_function
):

    per_query_metrics = []

    distribution = {
        "3/3": 0,
        "2/3": 0,
        "1/3": 0,
        "0/3": 0
    }

    for query_filename, query_data in (
        all_results.items()
    ):

        candidates = query_data[
            "candidates"
        ]

        # -----------------------------------------------
        # Rank candidates
        # -----------------------------------------------

        ranked = ranking_function(
            candidates
        )

        # -----------------------------------------------
        # Metrics
        # -----------------------------------------------

        metrics = evaluate_query(
            query_filename,
            ranked
        )

        per_query_metrics.append(
            metrics
        )

        # -----------------------------------------------
        # Top-10 distribution
        # -----------------------------------------------

        relevant10 = sum(
            is_relevant(
                query_filename,
                c["candidate_id"]
            )
            for c in ranked[:10]
        )

        relevant10 = min(
            relevant10,
            3
        )

        distribution[
            f"{relevant10}/3"
        ] += 1

    # ========================================================
    # AVERAGE METRICS
    # ========================================================

    number_queries = len(
        per_query_metrics
    )

    averages = {}

    metric_names = [
        "recall@1",
        "recall@4",
        "recall@10",

        "precision@1",
        "precision@4",
        "precision@10",

        "mrr",

        "hit@1",
        "hit@4",
        "hit@10",

        "relevant@4",
        "relevant@10"
    ]

    for metric in metric_names:

        averages[metric] = (
            sum(
                x[metric]
                for x in per_query_metrics
            )
            /
            number_queries
        )

    averages[
        "distribution"
    ] = distribution

    return averages


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD RAW RESULTS
    # ========================================================

    print("=" * 120)
    print("LOADING RAW RETRIEVAL RESULTS")
    print("=" * 120)

    with gzip.open(
        INPUT_PATH,
        "rb"
    ) as f:

        all_results = pickle.load(
            f
        )

    print(
        f"Queries loaded: "
        f"{len(all_results)}"
    )

    # ========================================================
    # DEFINE STRATEGIES
    # ========================================================

    strategies = {

        # ----------------------------------------------------
        # 1. Current baseline
        # ----------------------------------------------------

        "RANSAC inliers":
            lambda c:
                rank_direct(
                    c,
                    "ransac_inliers"
                ),

        # ----------------------------------------------------
        # 2. Inlier ratio
        # ----------------------------------------------------

        "Inlier ratio":
            lambda c:
                rank_direct(
                    c,
                    "inlier_ratio"
                ),

        # ----------------------------------------------------
        # 3. Inliers × ratio
        # ----------------------------------------------------

        "Inliers x ratio":
            lambda c:
                rank_inliers_x_ratio(
                    c
                ),

        # ----------------------------------------------------
        # 4. TF-IDF only
        # ----------------------------------------------------

        "TF-IDF":
            lambda c:
                rank_direct(
                    c,
                    "tfidf_score"
                ),

        # ----------------------------------------------------
        # 5. Good matches
        # ----------------------------------------------------

        "Good matches":
            lambda c:
                rank_direct(
                    c,
                    "good_matches"
                ),

        # ----------------------------------------------------
        # 6. TF-IDF + RANSAC
        # ----------------------------------------------------

        "TFIDF + RANSAC":
            lambda c:
                rank_weighted(
                    c,
                    tfidf_weight=0.5,
                    ransac_weight=0.5,
                    ratio_weight=0.0,
                    good_weight=0.0
                ),

        # ----------------------------------------------------
        # 7. TF-IDF + Ratio
        # ----------------------------------------------------

        "TFIDF + Ratio":
            lambda c:
                rank_weighted(
                    c,
                    tfidf_weight=0.5,
                    ransac_weight=0.0,
                    ratio_weight=0.5,
                    good_weight=0.0
                ),

        # ----------------------------------------------------
        # 8. RANSAC + Ratio
        # ----------------------------------------------------

        "RANSAC + Ratio":
            lambda c:
                rank_weighted(
                    c,
                    tfidf_weight=0.0,
                    ransac_weight=0.5,
                    ratio_weight=0.5,
                    good_weight=0.0
                ),

        # ----------------------------------------------------
        # 9. TF-IDF + RANSAC + Ratio
        # ----------------------------------------------------

        "TFIDF + RANSAC + Ratio":
            lambda c:
                rank_weighted(
                    c,
                    tfidf_weight=1/3,
                    ransac_weight=1/3,
                    ratio_weight=1/3,
                    good_weight=0.0
                ),

        # ----------------------------------------------------
        # 10. All four
        # ----------------------------------------------------

        "All four":
            lambda c:
                rank_weighted(
                    c,
                    tfidf_weight=0.25,
                    ransac_weight=0.25,
                    ratio_weight=0.25,
                    good_weight=0.25
                )
    }

    # ========================================================
    # EVALUATE
    # ========================================================

    all_metrics = {}

    for strategy_name, ranking_function in (
        strategies.items()
    ):

        print(
            f"\nEvaluating: "
            f"{strategy_name}"
        )

        metrics = evaluate_strategy(
            all_results,
            ranking_function
        )

        all_metrics[
            strategy_name
        ] = metrics

    # ========================================================
    # MAIN COMPARISON TABLE
    # ========================================================

    print()
    print("=" * 140)
    print("RANKING STRATEGY COMPARISON")
    print("=" * 140)

    print(
        f"{'Strategy':30}"
        f"{'R@1':>8}"
        f"{'R@4':>8}"
        f"{'R@10':>8}"
        f"{'P@1':>8}"
        f"{'P@4':>8}"
        f"{'P@10':>8}"
        f"{'MRR':>8}"
        f"{'H@1':>8}"
        f"{'H@4':>8}"
        f"{'H@10':>8}"
    )

    print("-" * 140)

    for strategy_name, m in (
        all_metrics.items()
    ):

        print(
            f"{strategy_name:30}"

            f"{m['recall@1']:8.4f}"

            f"{m['recall@4']:8.4f}"

            f"{m['recall@10']:8.4f}"

            f"{m['precision@1']:8.4f}"

            f"{m['precision@4']:8.4f}"

            f"{m['precision@10']:8.4f}"

            f"{m['mrr']:8.4f}"

            f"{m['hit@1']:8.4f}"

            f"{m['hit@4']:8.4f}"

            f"{m['hit@10']:8.4f}"
        )

    # ========================================================
    # ADDITIONAL METRICS
    # ========================================================

    print()
    print("=" * 100)
    print("ADDITIONAL METRICS")
    print("=" * 100)

    for strategy_name, m in (
        all_metrics.items()
    ):

        print()
        print(
            f"--- {strategy_name} ---"
        )

        print(
            f"Mean relevant Top-4  : "
            f"{m['relevant@4']:.4f}"
        )

        print(
            f"Mean relevant Top-10 : "
            f"{m['relevant@10']:.4f}"
        )

        print(
            "Top-10 distribution:"
        )

        total = len(
            all_results
        )

        for category in [
            "3/3",
            "2/3",
            "1/3",
            "0/3"
        ]:

            count = m[
                "distribution"
            ][category]

            percentage = (
                count / total * 100
            )

            print(
                f"  {category}: "
                f"{count:3d} "
                f"({percentage:6.2f}%)"
            )


if __name__ == "__main__":
    main()