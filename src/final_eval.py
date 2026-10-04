import os
import cv2
import numpy as np
import pickle
import time


FEATURES_PATH = "../results/sift_features.pkl"
TFIDF_PATH = "../results/tfidf_index_2000.pkl"

DATASET_PATH = "../dataset/ukbench"

CANDIDATE_COUNT = 574
TOP_K = 10

# Full UKBench dataset:
# 10,200 images / 4 images per group = 2,550 groups
# One query from every group
QUERY_INDICES = range(0, 10200, 4)


def extract_query_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    keypoint_locations = np.float32([
        kp.pt for kp in keypoints
    ])

    return keypoint_locations, descriptors


def get_good_matches(
    query_descriptors,
    database_descriptors
):

    if (
        query_descriptors is None
        or database_descriptors is None
    ):
        return []

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(
        query_descriptors,
        database_descriptors,
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches


def geometric_verification(
    query_keypoints,
    database_keypoints,
    good_matches
):

    if len(good_matches) < 4:
        return 0

    src_points = np.float32([
        query_keypoints[m.queryIdx]
        for m in good_matches
    ])

    dst_points = np.float32([
        database_keypoints[m.trainIdx]
        for m in good_matches
    ])

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )

    if H is None or mask is None:
        return 0

    return int(np.sum(mask))


def get_query_tfidf(
    query_descriptors,
    vocabulary,
    idf
):

    if query_descriptors is None:
        return None

    words = []

    batch_size = 500

    for i in range(
        0,
        len(query_descriptors),
        batch_size
    ):

        batch = query_descriptors[
            i:i + batch_size
        ]

        distances = np.linalg.norm(
            batch[:, None, :]
            -
            vocabulary[None, :, :],
            axis=2
        )

        batch_words = np.argmin(
            distances,
            axis=1
        )

        words.extend(batch_words)

    words = np.array(words)

    tf = np.bincount(
        words,
        minlength=len(idf)
    ).astype(np.float32)

    if tf.sum() > 0:
        tf /= tf.sum()

    tfidf = tf * idf

    norm = np.linalg.norm(tfidf)

    if norm > 0:
        tfidf /= norm

    return tfidf


def retrieve_candidates(
    query_descriptors,
    tfidf_index,
    vocabulary,
    candidate_count
):

    idf = tfidf_index["idf"]

    tfidf_vectors = tfidf_index[
        "tfidf_vectors"
    ]

    query_tfidf = get_query_tfidf(
        query_descriptors,
        vocabulary,
        idf
    )

    scores = []

    for filename, vector in tfidf_vectors.items():

        score = float(
            np.dot(
                query_tfidf,
                vector
            )
        )

        scores.append(
            (filename, score)
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:candidate_count]


def evaluate_query(
    query_index,
    features,
    tfidf_index,
    vocabulary
):

    query_filename = (
        f"ukbench{query_index:05d}.jpg"
    )

    query_path = os.path.join(
        DATASET_PATH,
        query_filename
    )

    query_keypoints, query_descriptors = (
        extract_query_features(query_path)
    )

    # =========================================================
    # STAGE 1: TF-IDF
    # =========================================================

    stage1_start = time.time()

    candidates = retrieve_candidates(
        query_descriptors,
        tfidf_index,
        vocabulary,
        CANDIDATE_COUNT
    )

    stage1_time = time.time() - stage1_start

    # =========================================================
    # STAGE 2: SIFT MATCHING + RANSAC
    #
    # IMPORTANT:
    # Query image is NOT removed here.
    # It stays in the ranking for N-S evaluation.
    # =========================================================

    stage2_start = time.time()

    verified_results = []

    for filename, tfidf_score in candidates:

        data = features[filename]

        good_matches = get_good_matches(
            query_descriptors,
            data["descriptors"]
        )

        inliers = geometric_verification(
            query_keypoints,
            data["keypoints"],
            good_matches
        )

        verified_results.append(
            (
                filename,
                inliers,
                tfidf_score,
                len(good_matches)
            )
        )

    # Rank by RANSAC inliers
    verified_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Keep query INCLUDED
    final_results = verified_results[:TOP_K]

    stage2_time = time.time() - stage2_start

    # =========================================================
    # GROUND TRUTH
    # =========================================================

    group = query_index // 4

    group_images = {
        f"ukbench{4 * group + i:05d}.jpg"
        for i in range(4)
    }

    # The 3 OTHER relevant images
    relevant = group_images - {
        query_filename
    }

    # =========================================================
    # FULL RANKING
    # Query is still included here
    # =========================================================

    ranked_filenames_full = [
        filename
        for filename, _, _, _
        in final_results
    ]

    # =========================================================
    # UKBENCH N-S
    #
    # Top 4 results INCLUDING the query.
    # Count how many belong to the query's
    # 4-image object group.
    # =========================================================

    top4_full = ranked_filenames_full[:4]

    ns_score = len(
        set(top4_full) & group_images
    )

    # =========================================================
    # RETRIEVAL METRICS
    #
    # Remove query for comparison with previous
    # 100/300-query evaluations.
    # =========================================================

    ranked_filenames = [
        filename
        for filename in ranked_filenames_full
        if filename != query_filename
    ]

    # ---------------------------------------------------------
    # Relevant images @ K
    # ---------------------------------------------------------

    def relevant_at_k(k):

        return len(
            set(ranked_filenames[:k])
            & relevant
        )

    correct_1 = relevant_at_k(1)
    correct_4 = relevant_at_k(4)
    correct_10 = relevant_at_k(10)

    # ---------------------------------------------------------
    # Recall
    # ---------------------------------------------------------

    recall_1 = correct_1 / 3
    recall_4 = correct_4 / 3
    recall_10 = correct_10 / 3

    # ---------------------------------------------------------
    # Precision
    # ---------------------------------------------------------

    precision_1 = correct_1 / 1
    precision_4 = correct_4 / 4
    precision_10 = correct_10 / 10

    # ---------------------------------------------------------
    # MRR
    # ---------------------------------------------------------

    reciprocal_rank = 0.0

    for rank, filename in enumerate(
        ranked_filenames,
        start=1
    ):

        if filename in relevant:

            reciprocal_rank = 1.0 / rank

            break

    # ---------------------------------------------------------
    # Hit@K
    # ---------------------------------------------------------

    hit_1 = int(correct_1 > 0)
    hit_4 = int(correct_4 > 0)
    hit_10 = int(correct_10 > 0)

    # ---------------------------------------------------------
    # Relevant images in Top-K
    # ---------------------------------------------------------

    relevant_in_top4 = correct_4
    relevant_in_top10 = correct_10

    # =========================================================
    # RETURN
    # =========================================================

    return {

        "query": query_filename,

        # N-S
        "ns_score": ns_score,

        # Existing retrieval metrics
        "correct": correct_10,

        "recall": recall_10,

        "recall@1": recall_1,
        "recall@4": recall_4,
        "recall@10": recall_10,

        "precision@1": precision_1,
        "precision@4": precision_4,
        "precision@10": precision_10,

        "hit@1": hit_1,
        "hit@4": hit_4,
        "hit@10": hit_10,

        "relevant@4": relevant_in_top4,
        "relevant@10": relevant_in_top10,

        "mrr": reciprocal_rank,

        # Runtime
        "stage1_time": stage1_time,
        "stage2_time": stage2_time,
        "total_time": (
            stage1_time
            + stage2_time
        ),

        # Top-10 ranking INCLUDING query
        "results": final_results,

        # Ground truth
        "query_group": sorted(group_images),
        "relevant_images": sorted(relevant)
    }


def main():

    # =========================================================
    # LOAD DATABASE
    # =========================================================

    print("Loading SIFT database...")

    with open(
        FEATURES_PATH,
        "rb"
    ) as f:

        features = pickle.load(f)

    print("SIFT database loaded.")

    # =========================================================
    # LOAD TF-IDF
    # =========================================================

    print("Loading TF-IDF index...")

    with open(
        TFIDF_PATH,
        "rb"
    ) as f:

        tfidf_index = pickle.load(f)

    print("TF-IDF index loaded.")

    # =========================================================
    # LOAD VOCABULARY
    # =========================================================

    print("Loading visual vocabulary...")

    vocabulary = np.load(
        "../results/visual_vocabulary_2000.npy"
    )

    print("Vocabulary loaded.")
    print(
        "Vocabulary shape:",
        vocabulary.shape
    )

    # =========================================================
    # EVALUATION
    # =========================================================

    all_results = []

    total_start = time.time()

    total_queries = len(
        list(QUERY_INDICES)
    )

    print(
        f"\nRunning evaluation on "
        f"{total_queries} queries..."
    )

    for count, query_index in enumerate(
        QUERY_INDICES,
        start=1
    ):

        print(
            f"\n[{count}/{total_queries}] "
            f"Query: "
            f"ukbench{query_index:05d}.jpg"
        )

        result = evaluate_query(
            query_index,
            features,
            tfidf_index,
            vocabulary
        )

        all_results.append(result)

        print(
            f"N-S: "
            f"{result['ns_score']}/4"
        )

        print(
            f"Recall@10: "
            f"{result['correct']}/3 "
            f"({result['recall']:.3f})"
        )

        print(
            f"Stage 1: "
            f"{result['stage1_time']:.3f}s"
        )

        print(
            f"Stage 2: "
            f"{result['stage2_time']:.3f}s"
        )

        print(
            f"Total: "
            f"{result['total_time']:.3f}s"
        )

    total_time = (
        time.time()
        - total_start
    )

    # =========================================================
    # OVERALL STATISTICS
    # =========================================================

    # ---------------------------------------------------------
    # N-S
    # ---------------------------------------------------------

    mean_ns = np.mean([
        r["ns_score"]
        for r in all_results
    ])

    # ---------------------------------------------------------
    # Recall
    # ---------------------------------------------------------

    mean_recall_1 = np.mean([
        r["recall@1"]
        for r in all_results
    ])

    mean_recall_4 = np.mean([
        r["recall@4"]
        for r in all_results
    ])

    mean_recall_10 = np.mean([
        r["recall@10"]
        for r in all_results
    ])

    # ---------------------------------------------------------
    # Precision
    # ---------------------------------------------------------

    mean_precision_1 = np.mean([
        r["precision@1"]
        for r in all_results
    ])

    mean_precision_4 = np.mean([
        r["precision@4"]
        for r in all_results
    ])

    mean_precision_10 = np.mean([
        r["precision@10"]
        for r in all_results
    ])

    # ---------------------------------------------------------
    # MRR
    # ---------------------------------------------------------

    mean_mrr = np.mean([
        r["mrr"]
        for r in all_results
    ])

    # ---------------------------------------------------------
    # Relevant images
    # ---------------------------------------------------------

    mean_relevant_4 = np.mean([
        r["relevant@4"]
        for r in all_results
    ])

    mean_relevant_10 = np.mean([
        r["relevant@10"]
        for r in all_results
    ])

    # ---------------------------------------------------------
    # Hit rate
    # ---------------------------------------------------------

    hit_rate_1 = np.mean([
        r["hit@1"]
        for r in all_results
    ])

    hit_rate_4 = np.mean([
        r["hit@4"]
        for r in all_results
    ])

    hit_rate_10 = np.mean([
        r["hit@10"]
        for r in all_results
    ])

    # ---------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------

    mean_stage1 = np.mean([
        r["stage1_time"]
        for r in all_results
    ])

    mean_stage2 = np.mean([
        r["stage2_time"]
        for r in all_results
    ])

    mean_total = np.mean([
        r["total_time"]
        for r in all_results
    ])

    # ---------------------------------------------------------
    # Top-10 distribution
    # ---------------------------------------------------------

    perfect = sum(
        r["correct"] == 3
        for r in all_results
    )

    two_correct = sum(
        r["correct"] == 2
        for r in all_results
    )

    one_correct = sum(
        r["correct"] == 1
        for r in all_results
    )

    zero_correct = sum(
        r["correct"] == 0
        for r in all_results
    )

    # ---------------------------------------------------------
    # Candidate reduction
    # ---------------------------------------------------------

    dataset_size = len(features)

    candidate_percentage = (
        CANDIDATE_COUNT
        / dataset_size
    ) * 100

    eliminated_percentage = (
        1
        -
        CANDIDATE_COUNT
        / dataset_size
    ) * 100

    # =========================================================
    # FINAL RESULTS
    # =========================================================

    print("\n" + "=" * 60)
    print("FINAL FULL UKBENCH EVALUATION")
    print("=" * 60)

    print(
        f"Dataset: UKBench"
    )

    print(
        f"Queries evaluated: "
        f"{total_queries}"
    )

    print(
        f"Images in dataset: "
        f"{dataset_size}"
    )

    print(
        f"Groups evaluated: "
        f"{total_queries}"
    )

    print(
        f"Relevant images per query: "
        f"3"
    )

    # =========================================================
    # N-S
    # =========================================================

    print("\n--- UKBench N-S ---")

    print(
        f"Mean N-S score: "
        f"{mean_ns:.4f} / 4"
    )

    print(
        f"Mean N-S percentage: "
        f"{mean_ns / 4 * 100:.2f}%"
    )

    # =========================================================
    # RETRIEVAL
    # =========================================================

    print("\n--- Retrieval Performance ---")

    print(
        f"Recall@1:  "
        f"{mean_recall_1:.4f} "
        f"({mean_recall_1 * 100:.2f}%)"
    )

    print(
        f"Recall@4:  "
        f"{mean_recall_4:.4f} "
        f"({mean_recall_4 * 100:.2f}%)"
    )

    print(
        f"Recall@10: "
        f"{mean_recall_10:.4f} "
        f"({mean_recall_10 * 100:.2f}%)"
    )

    print(
        f"Precision@1:  "
        f"{mean_precision_1:.4f}"
    )

    print(
        f"Precision@4:  "
        f"{mean_precision_4:.4f}"
    )

    print(
        f"Precision@10: "
        f"{mean_precision_10:.4f}"
    )

    print(
        f"MRR: "
        f"{mean_mrr:.4f}"
    )

    print(
        f"Mean relevant images in Top-4: "
        f"{mean_relevant_4:.4f} / 3"
    )

    print(
        f"Mean relevant images in Top-10: "
        f"{mean_relevant_10:.4f} / 3"
    )

    # =========================================================
    # HIT RATE
    # =========================================================

    print("\n--- Hit Rate ---")

    print(
        f"Hit@1:  "
        f"{hit_rate_1 * 100:.2f}%"
    )

    print(
        f"Hit@4:  "
        f"{hit_rate_4 * 100:.2f}%"
    )

    print(
        f"Hit@10: "
        f"{hit_rate_10 * 100:.2f}%"
    )

    # =========================================================
    # TOP-10 DISTRIBUTION
    # =========================================================

    print("\n--- Top-10 Distribution ---")

    print(
        f"3/3 relevant: "
        f"{perfect}/{total_queries} "
        f"({perfect / total_queries * 100:.2f}%)"
    )

    print(
        f"2/3 relevant: "
        f"{two_correct}/{total_queries} "
        f"({two_correct / total_queries * 100:.2f}%)"
    )

    print(
        f"1/3 relevant: "
        f"{one_correct}/{total_queries} "
        f"({one_correct / total_queries * 100:.2f}%)"
    )

    print(
        f"0/3 relevant: "
        f"{zero_correct}/{total_queries} "
        f"({zero_correct / total_queries * 100:.2f}%)"
    )

    # =========================================================
    # RUNTIME
    # =========================================================

    print("\n--- Runtime ---")

    print(
        f"Candidate count: "
        f"{CANDIDATE_COUNT}"
    )

    print(
        f"Candidate percentage: "
        f"{candidate_percentage:.2f}% "
        f"of dataset"
    )

    print(
        f"Images eliminated before Stage 2: "
        f"{eliminated_percentage:.2f}%"
    )

    print(
        f"Mean Stage 1 time: "
        f"{mean_stage1:.3f}s"
    )

    print(
        f"Mean Stage 2 time: "
        f"{mean_stage2:.3f}s"
    )

    print(
        f"Mean total/query: "
        f"{mean_total:.3f}s"
    )

    print(
        f"Total experiment time: "
        f"{total_time:.2f}s "
        f"({total_time / 60:.2f} min)"
    )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    output = {

        "configuration": {

            "dataset": "UKBench",

            "dataset_size":
                dataset_size,

            "queries":
                total_queries,

            "candidate_count":
                CANDIDATE_COUNT,

            "top_k":
                TOP_K,

            "relevant_per_query":
                3,

            "sift":
                True,

            "vocabulary_size":
                len(vocabulary),

            "ratio_test":
                0.75,

            "ransac_threshold":
                5.0,

            "query_indices":
                "range(0, 10200, 4)"
        },

        "metrics": {

            "N-S":
                mean_ns,

            "N-S_percentage":
                mean_ns / 4,

            "recall@1":
                mean_recall_1,

            "recall@4":
                mean_recall_4,

            "recall@10":
                mean_recall_10,

            "precision@1":
                mean_precision_1,

            "precision@4":
                mean_precision_4,

            "precision@10":
                mean_precision_10,

            "mrr":
                mean_mrr,

            "mean_relevant@4":
                mean_relevant_4,

            "mean_relevant@10":
                mean_relevant_10,

            "hit@1":
                hit_rate_1,

            "hit@4":
                hit_rate_4,

            "hit@10":
                hit_rate_10,

            "perfect_3of3":
                perfect,

            "correct_2of3":
                two_correct,

            "correct_1of3":
                one_correct,

            "correct_0of3":
                zero_correct,

            "candidate_percentage":
                candidate_percentage,

            "eliminated_percentage":
                eliminated_percentage,

            "mean_stage1_time":
                mean_stage1,

            "mean_stage2_time":
                mean_stage2,

            "mean_total_time":
                mean_total,

            "total_experiment_time":
                total_time
        },

        "queries":
            all_results
    }

    output_path = (
        "../results/final_eval.pkl"
    )

    with open(
        output_path,
        "wb"
    ) as f:

        pickle.dump(
            output,
            f
        )

    print(
        "\nSaved detailed results to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()