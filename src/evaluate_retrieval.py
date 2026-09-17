import cv2
import numpy as np
import pickle
import time
import os


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"
INDEX_PATH = "../results/tfidf_index_2000.pkl"

DATASET_PATH = "../dataset/ukbench"

NUM_QUERIES = 10
CANDIDATE_COUNT = 574
TOP_K = 10


# --------------------------------------------------
# Helper: determine relevant UKBench images
# --------------------------------------------------

def get_relevant_images(query_filename):

    query_number = int(
        query_filename
        .replace("ukbench", "")
        .replace(".jpg", "")
    )

    group_start = (query_number // 4) * 4

    relevant = {
        f"ukbench{group_start:05d}.jpg",
        f"ukbench{group_start + 1:05d}.jpg",
        f"ukbench{group_start + 2:05d}.jpg",
        f"ukbench{group_start + 3:05d}.jpg"
    }

    relevant.remove(query_filename)

    return relevant


# --------------------------------------------------
# Normalize values to 0-1 within candidate set
# --------------------------------------------------

def normalize_scores(values):

    values = np.asarray(values, dtype=np.float32)

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum == 0:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


# --------------------------------------------------
# Load database ONCE
# --------------------------------------------------

print("Loading SIFT database...")

with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

print("Database loaded.")

print("Loading vocabulary...")
vocabulary = np.load(VOCABULARY_PATH)

print("Loading TF-IDF index...")

with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)

tfidf_vectors = index["tfidf_vectors"]
idf = index["idf"]

print("Database:", len(tfidf_vectors))
print("Vocabulary:", len(vocabulary))


# --------------------------------------------------
# Create matchers ONCE
# --------------------------------------------------

bf_vocab = cv2.BFMatcher(cv2.NORM_L2)
bf = cv2.BFMatcher(cv2.NORM_L2)

sift = cv2.SIFT_create()


# --------------------------------------------------
# Scoring methods
# --------------------------------------------------

SCORING_METHODS = [
    "inliers",
    "good_matches",
    "inlier_ratio",
    "inliers_plus_good",
    "inliers_plus_tfidf",
    "good_plus_tfidf"
]


# Store Recall@10 for each scoring method
method_recalls = {
    method: []
    for method in SCORING_METHODS
}


# Store normal pipeline timing
query_results = []


# --------------------------------------------------
# Process queries
# --------------------------------------------------

for query_number in range(NUM_QUERIES):

    query_filename = f"ukbench{query_number:05d}.jpg"

    query_path = os.path.join(
        DATASET_PATH,
        query_filename
    )

    print("\n" + "=" * 70)
    print("Query:", query_filename)

    total_start = time.time()

    # --------------------------------------------------
    # Extract query SIFT
    # --------------------------------------------------

    image = cv2.imread(query_path)

    if image is None:
        print("Could not load image.")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    query_keypoints, query_descriptors = (
        sift.detectAndCompute(gray, None)
    )

    print(
        "Query descriptors:",
        len(query_descriptors)
    )

    # --------------------------------------------------
    # Stage 1: TF-IDF candidate retrieval
    # --------------------------------------------------

    stage1_start = time.time()

    matches = bf_vocab.match(
        query_descriptors,
        vocabulary
    )

    word_counts = {}

    for match in matches:

        word_id = match.trainIdx

        word_counts[word_id] = (
            word_counts.get(word_id, 0) + 1
        )

    # --------------------------------------------------
    # Build query TF-IDF vector
    # --------------------------------------------------

    query_vector = np.zeros(
        len(vocabulary),
        dtype=np.float32
    )

    total_words = sum(
        word_counts.values()
    )

    for word_id, count in word_counts.items():

        tf = count / total_words

        query_vector[word_id] = (
            tf * idf[word_id]
        )

    norm = np.linalg.norm(query_vector)

    if norm > 0:
        query_vector /= norm

    # --------------------------------------------------
    # Compare with all database vectors
    # --------------------------------------------------

    tfidf_scores = []

    for filename, vector in tfidf_vectors.items():

        if filename == query_filename:
            continue

        similarity = np.dot(
            query_vector,
            vector
        )

        tfidf_scores.append(
            (filename, float(similarity))
        )

    tfidf_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    candidates = tfidf_scores[:CANDIDATE_COUNT]

    stage1_end = time.time()

    stage1_time = (
        stage1_end - stage1_start
    )

    # --------------------------------------------------
    # Relevant images
    # --------------------------------------------------

    relevant = get_relevant_images(
        query_filename
    )

    # Find TF-IDF ranks
    relevant_ranks = {}

    for target in relevant:

        for rank, (filename, score) in enumerate(
            tfidf_scores,
            start=1
        ):

            if filename == target:

                relevant_ranks[target] = rank
                break

    # Candidate recall

    candidate_filenames = {
        filename
        for filename, _ in candidates
    }

    candidates_containing_relevant = len(
        relevant &
        candidate_filenames
    )

    candidate_recall = (
        candidates_containing_relevant / 3
    )

    # --------------------------------------------------
    # Stage 2: SIFT + Lowe + RANSAC
    # --------------------------------------------------

    stage2_start = time.time()

    verified_results = []

    for filename, tfidf_score in candidates:

        data = features[filename]

        database_descriptors = (
            data["descriptors"]
        )

        database_keypoints = (
            data["keypoints"]
        )

        if (
            query_descriptors is None
            or database_descriptors is None
        ):
            continue

        # ----------------------------------------------
        # KNN matching
        # ----------------------------------------------

        knn_matches = bf.knnMatch(
            query_descriptors,
            database_descriptors,
            k=2
        )

        # ----------------------------------------------
        # Lowe ratio test
        # ----------------------------------------------

        good_matches = []

        for m, n in knn_matches:

            if m.distance < 0.75 * n.distance:

                good_matches.append(m)

        good_match_count = len(good_matches)

        # ----------------------------------------------
        # RANSAC
        # ----------------------------------------------

        inliers = 0

        if good_match_count >= 4:

            src_points = np.float32([
                query_keypoints[m.queryIdx].pt
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

            if (
                H is not None
                and mask is not None
            ):

                inliers = int(
                    np.sum(mask)
                )

        # ----------------------------------------------
        # Inlier ratio
        # ----------------------------------------------

        if good_match_count > 0:

            inlier_ratio = (
                inliers /
                good_match_count
            )

        else:

            inlier_ratio = 0.0

        verified_results.append({
            "filename": filename,
            "inliers": inliers,
            "good_matches": good_match_count,
            "inlier_ratio": inlier_ratio,
            "tfidf": tfidf_score
        })

    stage2_end = time.time()

    stage2_time = (
        stage2_end - stage2_start
    )

    # --------------------------------------------------
    # Extract arrays for normalization
    # --------------------------------------------------

    inliers_array = np.array([
        r["inliers"]
        for r in verified_results
    ])

    good_matches_array = np.array([
        r["good_matches"]
        for r in verified_results
    ])

    inlier_ratio_array = np.array([
        r["inlier_ratio"]
        for r in verified_results
    ])

    tfidf_array = np.array([
        r["tfidf"]
        for r in verified_results
    ])

    # Normalize each score within candidates

    normalized_inliers = normalize_scores(
        inliers_array
    )

    normalized_good_matches = normalize_scores(
        good_matches_array
    )

    normalized_inlier_ratio = normalize_scores(
        inlier_ratio_array
    )

    normalized_tfidf = normalize_scores(
        tfidf_array
    )

    # --------------------------------------------------
    # Add all experimental scores
    # --------------------------------------------------

    for i, result in enumerate(verified_results):

        result["score_inliers"] = (
            normalized_inliers[i]
        )

        result["score_good_matches"] = (
            normalized_good_matches[i]
        )

        result["score_inlier_ratio"] = (
            normalized_inlier_ratio[i]
        )

        result["score_inliers_plus_good"] = (
            normalized_inliers[i]
            + normalized_good_matches[i]
        )

        result["score_inliers_plus_tfidf"] = (
            normalized_inliers[i]
            + normalized_tfidf[i]
        )

        result["score_good_plus_tfidf"] = (
            normalized_good_matches[i]
            + normalized_tfidf[i]
        )

    # --------------------------------------------------
    # Evaluate every scoring method
    # --------------------------------------------------

    method_top10 = {}

    for method in SCORING_METHODS:

        score_key = "score_" + method

        ranked_results = sorted(
            verified_results,
            key=lambda x: x[score_key],
            reverse=True
        )

        top10 = [
            result["filename"]
            for result in ranked_results[:TOP_K]
        ]

        correct = len(
            set(top10) & relevant
        )

        recall = correct / 3

        method_recalls[method].append(
            recall
        )

        method_top10[method] = (
            correct,
            top10
        )

    total_end = time.time()

    total_time = (
        total_end - total_start
    )

    # --------------------------------------------------
    # Print query information
    # --------------------------------------------------

    print(
        "\nRelevant TF-IDF ranks:"
    )

    for filename, rank in relevant_ranks.items():

        print(
            f"  {filename}: rank {rank}"
        )

    print(
        "\nCandidate Recall:",
        f"{candidates_containing_relevant}/3"
    )

    print(
        "Stage 1:",
        f"{stage1_time:.3f}s"
    )

    print(
        "Stage 2:",
        f"{stage2_time:.3f}s"
    )

    print(
        "Total:",
        f"{total_time:.3f}s"
    )

    print(
        "\nRecall@10 by scoring method:"
    )

    for method in SCORING_METHODS:

        correct, _ = method_top10[method]

        print(
            f"  {method:25s}: "
            f"{correct}/3"
        )

    # --------------------------------------------------
    # Store timing result
    # --------------------------------------------------

    query_results.append({
        "query": query_filename,
        "descriptors": len(query_descriptors),
        "candidate_recall": candidate_recall,
        "stage1_time": stage1_time,
        "stage2_time": stage2_time,
        "total_time": total_time
    })


# ======================================================
# Overall evaluation
# ======================================================

print("\n\n")
print("=" * 70)
print("OVERALL SCORING RESULTS")
print("=" * 70)

print(
    f"\nQueries evaluated: {len(query_results)}"
)

print(
    "\nAverage Recall@10:"
)

for method in SCORING_METHODS:

    if len(method_recalls[method]) > 0:

        average_recall = np.mean(
            method_recalls[method]
        )

        print(
            f"  {method:25s}: "
            f"{average_recall:.3f} "
            f"({average_recall * 100:.1f}%)"
        )


# --------------------------------------------------
# Timing results
# --------------------------------------------------

if len(query_results) > 0:

    average_candidate_recall = np.mean([
        r["candidate_recall"]
        for r in query_results
    ])

    average_stage1 = np.mean([
        r["stage1_time"]
        for r in query_results
    ])

    average_stage2 = np.mean([
        r["stage2_time"]
        for r in query_results
    ])

    average_total = np.mean([
        r["total_time"]
        for r in query_results
    ])

    print(
        "\nAverage Candidate Recall:",
        f"{average_candidate_recall:.3f}"
    )

    print(
        "Average Stage 1 time:",
        f"{average_stage1:.3f}s"
    )

    print(
        "Average Stage 2 time:",
        f"{average_stage2:.3f}s"
    )

    print(
        "Average Total time:",
        f"{average_total:.3f}s"
    )