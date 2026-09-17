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

# One query from each of the first 100 UKBench groups
QUERY_INDICES = range(0, 400, 4)


def extract_query_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    keypoint_locations = np.float32([kp.pt for kp in keypoints])

    return keypoint_locations, descriptors


def get_good_matches(query_descriptors, database_descriptors):
    if query_descriptors is None or database_descriptors is None:
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

    # Compute nearest visual word in batches
    words = []

    batch_size = 500

    for i in range(0, len(query_descriptors), batch_size):

        batch = query_descriptors[i:i + batch_size]

        distances = np.linalg.norm(
            batch[:, None, :] -
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
    tfidf_vectors = tfidf_index["tfidf_vectors"]

    query_tfidf = get_query_tfidf(
        query_descriptors,
        vocabulary,
        idf
    )

    scores = []

    for filename, vector in tfidf_vectors.items():

        score = float(
            np.dot(query_tfidf, vector)
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
    query_filename = f"ukbench{query_index:05d}.jpg"
    query_path = os.path.join(
        DATASET_PATH,
        query_filename
    )

    query_keypoints, query_descriptors = extract_query_features(
        query_path
    )

    # -------------------------
    # Stage 1: TF-IDF
    # -------------------------

    stage1_start = time.time()

    candidates = retrieve_candidates(
        query_descriptors,
        tfidf_index,
        vocabulary,
        CANDIDATE_COUNT
    )

    stage1_time = time.time() - stage1_start

    # -------------------------
    # Stage 2: RANSAC
    # -------------------------

    stage2_start = time.time()

    verified_results = []

    for filename, tfidf_score in candidates:

        if filename == query_filename:
            continue

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

    verified_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    final_results = verified_results[:TOP_K]

    stage2_time = time.time() - stage2_start

    # -------------------------
    # Ground truth
    # -------------------------

    group = query_index // 4

    relevant = {
        f"ukbench{4 * group + i:05d}.jpg"
        for i in range(4)
        if 4 * group + i != query_index
    }

    retrieved = {
        filename
        for filename, _, _, _ in final_results
    }

    correct = len(relevant & retrieved)

    return {
        "query": query_filename,
        "candidate_recall": None,
        "correct": correct,
        "recall": correct / 3,
        "stage1_time": stage1_time,
        "stage2_time": stage2_time,
        "total_time": stage1_time + stage2_time,
        "results": final_results
    }


def main():

    print("Loading SIFT database...")

    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    print("SIFT database loaded.")

    print("Loading TF-IDF index...")

    with open(TFIDF_PATH, "rb") as f:
        tfidf_index = pickle.load(f)

    print("TF-IDF index loaded.")

    print("Loading visual vocabulary...")

    # The vocabulary is stored separately.
    print("Loading visual vocabulary...")

    vocabulary = np.load(
        "../results/visual_vocabulary_2000.npy"
    )

    print("Vocabulary loaded.")
    print("Vocabulary shape:", vocabulary.shape)

    all_results = []

    total_start = time.time()

    for count, query_index in enumerate(QUERY_INDICES, start=1):

        print(
            f"\n[{count}/100] "
            f"Query: ukbench{query_index:05d}.jpg"
        )

        result = evaluate_query(
            query_index,
            features,
            tfidf_index,
            vocabulary
        )

        all_results.append(result)

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

    total_time = time.time() - total_start

    # -------------------------
    # Overall statistics
    # -------------------------

    mean_recall = np.mean([
        r["recall"]
        for r in all_results
    ])

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

    print("\n" + "=" * 50)
    print("FINAL 100-QUERY RESULTS")
    print("=" * 50)

    print(
        f"Candidate count: {CANDIDATE_COUNT}"
    )

    print(
        f"Mean Recall@10: "
        f"{mean_recall:.4f}"
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
        f"Mean total time: "
        f"{mean_total:.3f}s"
    )

    print(
        f"Total experiment time: "
        f"{total_time:.2f}s"
    )

    print(
        f"Queries with 3/3: "
        f"{sum(r['correct'] == 3 for r in all_results)}/100"
    )

    print(
        f"Queries with 2/3: "
        f"{sum(r['correct'] == 2 for r in all_results)}/100"
    )

    print(
        f"Queries with 1/3: "
        f"{sum(r['correct'] == 1 for r in all_results)}/100"
    )

    print(
        f"Queries with 0/3: "
        f"{sum(r['correct'] == 0 for r in all_results)}/100"
    )

    # Save detailed results
    with open(
        "../results/evaluation_100_574.pkl",
        "wb"
    ) as f:
        pickle.dump(all_results, f)

    print(
        "\nSaved detailed results to "
        "../results/evaluation_100_574.pkl"
    )


if __name__ == "__main__":
    main()