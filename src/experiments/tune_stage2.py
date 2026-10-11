import cv2
import numpy as np
import pickle
import time
import gc
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_4000.npy"
INDEX_PATH = "../results/tfidf_index_4000.pkl"


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_COUNT = 574

QUERY_INDICES = range(0, 1200, 4)

RATIO_THRESHOLDS = [
    0.65,
    0.70,
    0.75,
    0.80,
    0.85
]

RANSAC_THRESHOLDS = [
    3.0,
    5.0,
    7.0,
    10.0
]


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_scores(tfidf, good_matches, inliers):

    tfidf = np.asarray(tfidf, dtype=np.float64)
    good_matches = np.asarray(good_matches, dtype=np.float64)
    inliers = np.asarray(inliers, dtype=np.float64)

    def minmax(x):
        xmin = np.min(x)
        xmax = np.max(x)

        if xmax == xmin:
            return np.zeros_like(x)

        return (x - xmin) / (xmax - xmin)

    tfidf_norm = minmax(tfidf)
    good_norm = minmax(good_matches)
    inlier_norm = minmax(inliers)

    return (
        0.25 * tfidf_norm
        + 0.70 * np.log1p(good_norm)
        + 0.05 * np.log1p(inlier_norm)
    )


# ============================================================
# VISUAL SEARCH ENGINE
# ============================================================

class VisualSearchEngine:

    def __init__(self):

        print("Loading precomputed data...")

        with open(FEATURES_PATH, "rb") as f:
            self.features = pickle.load(f)

        self.vocabulary = np.load(VOCABULARY_PATH)

        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)

        # IMPORTANT:
        # Your index uses these keys.
        self.tfidf_vectors = index["tfidf_vectors"]
        self.idf = index["idf"]

        self.sift = cv2.SIFT_create()

        self.bf_vocab = cv2.BFMatcher(cv2.NORM_L2)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

        print("Engine initialized.")


    # ========================================================
    # STAGE 1
    # ========================================================

    def retrieve_candidates(self, query_descriptors, query_filename):

        # -----------------------------------------------
        # Assign query descriptors to visual words
        # -----------------------------------------------

        matches = self.bf_vocab.match(
            query_descriptors,
            self.vocabulary
        )

        word_counts = np.zeros(
            len(self.vocabulary),
            dtype=np.float32
        )

        for m in matches:
            word_counts[m.trainIdx] += 1

        # -----------------------------------------------
        # TF-IDF query vector
        # -----------------------------------------------

        query_vector = word_counts * self.idf

        norm = np.linalg.norm(query_vector)

        if norm > 0:
            query_vector /= norm

        # -----------------------------------------------
        # Compare against database
        # -----------------------------------------------

        scores = []

        for filename, vector in self.tfidf_vectors.items():

            if filename == query_filename:
                continue

            similarity = np.dot(
                query_vector,
                vector
            )

            scores.append(
                (
                    filename,
                    float(similarity)
                )
            )

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scores[:CANDIDATE_COUNT]


    # ========================================================
    # BUILD KNN FOR ONE QUERY
    # ========================================================

    def build_query_cache(
        self,
        query_filename,
        query_keypoints,
        query_descriptors
    ):

        candidates = self.retrieve_candidates(
            query_descriptors,
            query_filename
        )

        print(
            f"  Candidates: {len(candidates)}",
            flush=True
        )

        cache = []

        for candidate_number, (
            filename,
            tfidf_score
        ) in enumerate(candidates):

            data = self.features[filename]

            database_descriptors = data["descriptors"]
            database_keypoints = data["keypoints"]

            if database_descriptors is None:
                cache.append({
                    "candidate_id": filename,
                    "tfidf_score": float(tfidf_score),
                    "database_keypoints": database_keypoints,
                    "knn_matches": []
                })

                continue

            # -------------------------------------------
            # KNN IS DONE ONLY ONCE
            # -------------------------------------------

            knn_matches = self.bf.knnMatch(
                query_descriptors,
                database_descriptors,
                k=2
            )

            # -------------------------------------------
            # Store only this query's cache
            # -------------------------------------------

            cache.append({
                "candidate_id": filename,
                "tfidf_score": float(tfidf_score),
                "database_keypoints": database_keypoints,
                "knn_matches": knn_matches
            })

            if (candidate_number + 1) % 100 == 0:
                print(
                    f"  KNN: {candidate_number + 1}/{len(candidates)}",
                    flush=True
                )

        return cache


    # ========================================================
    # EVALUATE ONE CONFIGURATION FOR ONE QUERY
    # ========================================================

    def evaluate_query_configuration(
        self,
        cache,
        query_keypoints,
        ratio_threshold,
        ransac_threshold
    ):

        candidate_ids = []
        tfidf_scores = []
        good_match_counts = []
        ransac_inliers = []

        # -----------------------------------------------
        # Lowe ratio + RANSAC
        # -----------------------------------------------

        for item in cache:

            candidate_id = item["candidate_id"]
            tfidf_score = item["tfidf_score"]
            database_keypoints = item["database_keypoints"]
            knn_matches = item["knn_matches"]

            good_matches = []

            # -------------------------------------------
            # Lowe ratio test
            # -------------------------------------------

            for pair in knn_matches:

                if len(pair) < 2:
                    continue

                m, n = pair

                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

            good_count = len(good_matches)

            # -------------------------------------------
            # RANSAC
            # -------------------------------------------

            inliers = 0

            if good_count >= 4:

                src_points = np.float32([
                    query_keypoints[m.queryIdx].pt
                    for m in good_matches
                ])

                dst_points = np.float32([
                    database_keypoints[m.trainIdx]
                    for m in good_matches
                ])

                try:

                    H, mask = cv2.findHomography(
                        src_points,
                        dst_points,
                        cv2.RANSAC,
                        ransac_threshold
                    )

                    if H is not None and mask is not None:
                        inliers = int(np.sum(mask))

                except cv2.error:
                    inliers = 0

            candidate_ids.append(candidate_id)
            tfidf_scores.append(tfidf_score)
            good_match_counts.append(good_count)
            ransac_inliers.append(inliers)

        # -----------------------------------------------
        # Final nonlinear ranking
        # -----------------------------------------------

        final_scores = calculate_final_scores(
            tfidf_scores,
            good_match_counts,
            ransac_inliers
        )

        ranked_indices = np.argsort(
            final_scores
        )[::-1]

        ranked_ids = [
            candidate_ids[i]
            for i in ranked_indices
        ]

        return ranked_ids


# ============================================================
# METRICS
# ============================================================

def get_relevant_images(query_index):

    group_start = (query_index // 4) * 4

    relevant = set()

    for i in range(group_start, group_start + 4):

        if i != query_index:
            relevant.add(
                f"ukbench{i:05d}.jpg"
            )

    return relevant


def calculate_metrics(ranked_ids, query_index):

    relevant = get_relevant_images(query_index)

    top1 = ranked_ids[:1]
    top4 = ranked_ids[:4]
    top10 = ranked_ids[:10]

    # -----------------------------------------------
    # Number of relevant images
    # -----------------------------------------------

    rel1 = sum(x in relevant for x in top1)
    rel4 = sum(x in relevant for x in top4)
    rel10 = sum(x in relevant for x in top10)

    # -----------------------------------------------
    # UKBench N-S score
    #
    # Rank 1 = 4
    # Rank 2 = 3
    # Rank 3 = 2
    # Rank 4 = 1
    # -----------------------------------------------

    ns_score = 0

    weights = [4, 3, 2, 1]

    for i, candidate in enumerate(top4):

        if candidate in relevant:
            ns_score += weights[i]

    # -----------------------------------------------
    # Recall
    # -----------------------------------------------

    recall1 = rel1 / 3
    recall4 = rel4 / 3
    recall10 = rel10 / 3

    # -----------------------------------------------
    # Precision
    # -----------------------------------------------

    precision1 = rel1 / 1
    precision4 = rel4 / 4
    precision10 = rel10 / 10

    # -----------------------------------------------
    # MRR
    # -----------------------------------------------

    reciprocal_rank = 0.0

    for rank, candidate in enumerate(ranked_ids, start=1):

        if candidate in relevant:
            reciprocal_rank = 1.0 / rank
            break

    # -----------------------------------------------
    # Hit
    # -----------------------------------------------

    hit1 = 1 if rel1 > 0 else 0
    hit4 = 1 if rel4 > 0 else 0
    hit10 = 1 if rel10 > 0 else 0

    return {
        "ns": ns_score,
        "recall1": recall1,
        "recall4": recall4,
        "recall10": recall10,
        "precision1": precision1,
        "precision4": precision4,
        "precision10": precision10,
        "mrr": reciprocal_rank,
        "hit1": hit1,
        "hit4": hit4,
        "hit10": hit10,
        "rel4": rel4,
        "rel10": rel10
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UKBENCH STAGE-2 PARAMETER TUNING")
    print("=" * 70)

    print(f"Queries: {len(QUERY_INDICES)}")
    print(f"Candidates: {CANDIDATE_COUNT}")
    print(f"K: 4000")
    print(f"Configurations: {len(RATIO_THRESHOLDS) * len(RANSAC_THRESHOLDS)}")
    print()
    print("Lowe ratios:", RATIO_THRESHOLDS)
    print("RANSAC thresholds:", RANSAC_THRESHOLDS)
    print("=" * 70)

    engine = VisualSearchEngine()

    # --------------------------------------------------------
    # Create one accumulator for every configuration
    # --------------------------------------------------------

    accumulators = {}

    for ratio in RATIO_THRESHOLDS:

        for ransac in RANSAC_THRESHOLDS:

            key = (ratio, ransac)

            accumulators[key] = {
                "ns": 0.0,
                "recall1": 0.0,
                "recall4": 0.0,
                "recall10": 0.0,
                "precision1": 0.0,
                "precision4": 0.0,
                "precision10": 0.0,
                "mrr": 0.0,
                "hit1": 0.0,
                "hit4": 0.0,
                "hit10": 0.0,
                "rel4": 0.0,
                "rel10": 0.0
            }

    overall_start = time.time()

    # ========================================================
    # PROCESS ONE QUERY AT A TIME
    # ========================================================

    for query_number, query_index in enumerate(
        QUERY_INDICES,
        start=1
    ):

        query_filename = (
            f"ukbench{query_index:05d}.jpg"
        )

        print(
            f"\nQuery {query_number}/{len(QUERY_INDICES)}: "
            f"{query_filename}",
            flush=True
        )

        query_path = (
            f"../dataset/ukbench/"
            f"{query_filename}"
        )

        # ----------------------------------------------------
        # SIFT
        # ----------------------------------------------------

        image = cv2.imread(query_path)

        if image is None:
            print(
                f"WARNING: Could not load {query_path}",
                flush=True
            )
            continue

        query_keypoints, query_descriptors = (
            engine.sift.detectAndCompute(
                image,
                None
            )
        )

        if query_descriptors is None:
            print(
                "WARNING: No descriptors.",
                flush=True
            )
            continue

        # ----------------------------------------------------
        # KNN CACHE FOR THIS QUERY ONLY
        # ----------------------------------------------------

        cache = engine.build_query_cache(
            query_filename,
            query_keypoints,
            query_descriptors
        )

        # ----------------------------------------------------
        # Evaluate all 20 configurations immediately
        # ----------------------------------------------------

        for ratio in RATIO_THRESHOLDS:

            for ransac in RANSAC_THRESHOLDS:

                ranked_ids = engine.evaluate_query_configuration(
                    cache,
                    query_keypoints,
                    ratio,
                    ransac
                )

                metrics = calculate_metrics(
                    ranked_ids,
                    query_index
                )

                accumulator = accumulators[
                    (ratio, ransac)
                ]

                for key in accumulator:
                    accumulator[key] += metrics[key]

        # ----------------------------------------------------
        # IMPORTANT:
        # Destroy this query's cache before next query.
        # ----------------------------------------------------

        del cache
        del image
        del query_keypoints
        del query_descriptors

        # Periodic garbage collection
        if query_number % 10 == 0:
            gc.collect()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = time.time() - overall_start

        print(
            f"  Completed {query_number}/{len(QUERY_INDICES)}"
            f" | elapsed: {elapsed / 60:.1f} min",
            flush=True
        )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    num_queries = len(QUERY_INDICES)

    results = []

    for (ratio, ransac), acc in accumulators.items():

        result = {
            "ratio": ratio,
            "ransac": ransac,

            "ns": acc["ns"] / num_queries,
            "ns_percent": (
                acc["ns"] / num_queries
            ) / 4 * 100,

            "recall1": (
                acc["recall1"] / num_queries
            ),
            "recall4": (
                acc["recall4"] / num_queries
            ),
            "recall10": (
                acc["recall10"] / num_queries
            ),

            "precision1": (
                acc["precision1"] / num_queries
            ),
            "precision4": (
                acc["precision4"] / num_queries
            ),
            "precision10": (
                acc["precision10"] / num_queries
            ),

            "mrr": (
                acc["mrr"] / num_queries
            ),

            "hit1": (
                acc["hit1"] / num_queries
            ),
            "hit4": (
                acc["hit4"] / num_queries
            ),
            "hit10": (
                acc["hit10"] / num_queries
            ),

            "mean_rel4": (
                acc["rel4"] / num_queries
            ),
            "mean_rel10": (
                acc["rel10"] / num_queries
            )
        }

        results.append(result)


    # ========================================================
    # SORT BY N-S
    # ========================================================

    results.sort(
        key=lambda x: x["ns"],
        reverse=True
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("PARAMETER TUNING RESULTS")
    print("=" * 100)

    print(
        f"{'Ratio':<8}"
        f"{'RANSAC':<10}"
        f"{'N-S':<10}"
        f"{'N-S %':<10}"
        f"{'R@1':<10}"
        f"{'R@4':<10}"
        f"{'R@10':<10}"
        f"{'MRR':<10}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['ratio']:<8.2f}"
            f"{r['ransac']:<10.1f}"
            f"{r['ns']:<10.4f}"
            f"{r['ns_percent']:<10.2f}"
            f"{r['recall1'] * 100:<10.2f}"
            f"{r['recall4'] * 100:<10.2f}"
            f"{r['recall10'] * 100:<10.2f}"
            f"{r['mrr'] * 100:<10.2f}"
        )


    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    best = results[0]

    print("\n")
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    print(
        f"Lowe ratio:       {best['ratio']}"
    )

    print(
        f"RANSAC threshold: {best['ransac']}"
    )

    print(
        f"N-S:              "
        f"{best['ns']:.4f} / 4"
    )

    print(
        f"N-S percentage:   "
        f"{best['ns_percent']:.2f}%"
    )

    print(
        f"Recall@1:         "
        f"{best['recall1'] * 100:.2f}%"
    )

    print(
        f"Recall@4:         "
        f"{best['recall4'] * 100:.2f}%"
    )

    print(
        f"Recall@10:        "
        f"{best['recall10'] * 100:.2f}%"
    )

    print(
        f"Precision@1:      "
        f"{best['precision1']:.4f}"
    )

    print(
        f"Precision@4:      "
        f"{best['precision4']:.4f}"
    )

    print(
        f"Precision@10:     "
        f"{best['precision10']:.4f}"
    )

    print(
        f"MRR:              "
        f"{best['mrr'] * 100:.2f}%"
    )

    print(
        f"Hit@1:            "
        f"{best['hit1'] * 100:.2f}%"
    )

    print(
        f"Hit@4:            "
        f"{best['hit4'] * 100:.2f}%"
    )

    print(
        f"Hit@10:           "
        f"{best['hit10'] * 100:.2f}%"
    )

    print(
        f"Mean relevant Top-4:  "
        f"{best['mean_rel4']:.4f} / 3"
    )

    print(
        f"Mean relevant Top-10: "
        f"{best['mean_rel10']:.4f} / 3"
    )

    total_time = time.time() - overall_start

    print("\n")
    print(
        f"Total time: {total_time / 60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()