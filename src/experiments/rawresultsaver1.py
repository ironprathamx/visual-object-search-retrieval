import cv2
import numpy as np
import pickle
import gzip
import time
import os
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_4000.npy"
INDEX_PATH = "../results/tfidf_index_4000.pkl"

OUTPUT_PATH = "../results/retrieval_raw_300_4k.pkl.gz"


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_COUNT = 574

RATIO_THRESHOLD = 0.75

RANSAC_THRESHOLD = 5.0

# 300 evenly distributed queries.
#
# UKBench:
# 10200 images
# 2550 groups
# 4 images per group
#
# Every 4th image gives one query from each group.
QUERY_INDICES = range(0, 1200, 4)


# ============================================================
# VISUAL SEARCH ENGINE
# ============================================================

class VisualSearchEngine:

    def __init__(self):

        print("Loading precomputed data...")

        # ----------------------------------------------------
        # Load SIFT features
        # ----------------------------------------------------

        with open(FEATURES_PATH, "rb") as f:
            self.features = pickle.load(f)

        # ----------------------------------------------------
        # Load visual vocabulary
        # ----------------------------------------------------

        self.vocabulary = np.load(
            VOCABULARY_PATH
        )

        # ----------------------------------------------------
        # Load TF-IDF index
        # ----------------------------------------------------

        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)

        self.tfidf_vectors = index["tfidf_vectors"]
        self.idf = index["idf"]

        # ----------------------------------------------------
        # Initialize SIFT
        # ----------------------------------------------------

        self.sift = cv2.SIFT_create()

        # ----------------------------------------------------
        # Matcher for visual vocabulary
        # ----------------------------------------------------

        self.bf_vocab = cv2.BFMatcher(
            cv2.NORM_L2
        )

        # ----------------------------------------------------
        # Matcher for database descriptors
        # ----------------------------------------------------

        self.bf = cv2.BFMatcher(
            cv2.NORM_L2
        )

        print("Engine initialized.")

    # ========================================================
    # RAW SEARCH
    # ========================================================

    def search_raw(self, query_path):

        # ----------------------------------------------------
        # Load query image
        # ----------------------------------------------------

        image = cv2.imread(query_path)

        if image is None:
            raise ValueError(
                f"Could not read image: {query_path}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        total_start = time.time()

        # ----------------------------------------------------
        # SIFT feature extraction
        # ----------------------------------------------------

        query_keypoints, query_descriptors = (
            self.sift.detectAndCompute(
                gray,
                None
            )
        )

        if query_descriptors is None:
            raise ValueError(
                "No SIFT features detected in query image."
            )

        query_filename = os.path.basename(
            query_path
        )

        # ----------------------------------------------------
        # Assign query descriptors to visual words
        # ----------------------------------------------------

        matches = self.bf_vocab.match(
            query_descriptors,
            self.vocabulary
        )

        word_counts = {}

        for match in matches:

            word_id = match.trainIdx

            word_counts[word_id] = (
                word_counts.get(word_id, 0) + 1
            )

        # ----------------------------------------------------
        # Build query TF-IDF vector
        # ----------------------------------------------------

        query_vector = np.zeros(
            len(self.vocabulary),
            dtype=np.float32
        )

        total_words = sum(
            word_counts.values()
        )

        if total_words == 0:
            raise ValueError(
                "Query produced zero visual words."
            )

        for word_id, count in word_counts.items():

            tf = count / total_words

            query_vector[word_id] = (
                tf * self.idf[word_id]
            )

        # ----------------------------------------------------
        # Normalize query vector
        # ----------------------------------------------------

        norm = np.linalg.norm(
            query_vector
        )

        if norm > 0:
            query_vector /= norm

        # ====================================================
        # STAGE 1
        # TF-IDF CANDIDATE RETRIEVAL
        # ====================================================

        scores = []

        for filename, vector in self.tfidf_vectors.items():

            # Never compare query against itself.
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

        candidates = scores[
            :CANDIDATE_COUNT
        ]

        stage1_time = (
            time.time() - total_start
        )

        # ====================================================
        # STAGE 2
        # SIFT KNN + RATIO + RANSAC
        # ====================================================

        raw_results = []

        stage2_start = time.time()

        for candidate_number, (
            filename,
            tfidf_score
        ) in enumerate(candidates):

            data = self.features[filename]

            database_descriptors = (
                data["descriptors"]
            )

            database_keypoints = (
                data["keypoints"]
            )

            # ------------------------------------------------
            # Handle missing descriptors
            # ------------------------------------------------

            if database_descriptors is None:

                raw_results.append({
                    "candidate_id": filename,
                    "tfidf_score": tfidf_score,
                    "good_matches": 0,
                    "ransac_inliers": 0,
                    "inlier_ratio": 0.0
                })

                continue

            # =================================================
            # KNN MATCHING
            # =================================================

            knn_matches = self.bf.knnMatch(
                query_descriptors,
                database_descriptors,
                k=2
            )

            # =================================================
            # LOWE RATIO TEST
            # =================================================

            good_matches = []

            for pair in knn_matches:

                # Sometimes a descriptor may not have
                # two valid neighbors.
                if len(pair) < 2:
                    continue

                m, n = pair

                if (
                    m.distance
                    <
                    RATIO_THRESHOLD * n.distance
                ):
                    good_matches.append(m)

            good_match_count = len(
                good_matches
            )

            # =================================================
            # RANSAC
            # =================================================

            ransac_inliers = 0

            if good_match_count >= 4:

                src_points = np.float32([
                    query_keypoints[
                        m.queryIdx
                    ].pt
                    for m in good_matches
                ])

                dst_points = np.float32([
                    database_keypoints[
                        m.trainIdx
                    ]
                    for m in good_matches
                ])

                H, mask = cv2.findHomography(
                    src_points,
                    dst_points,
                    cv2.RANSAC,
                    RANSAC_THRESHOLD
                )

                if (
                    H is not None
                    and mask is not None
                ):

                    ransac_inliers = int(
                        np.sum(mask)
                    )

            # =================================================
            # INLIER RATIO
            # =================================================

            if good_match_count > 0:

                inlier_ratio = (
                    ransac_inliers
                    /
                    good_match_count
                )

            else:

                inlier_ratio = 0.0

            # =================================================
            # SAVE RAW CANDIDATE DATA
            # =================================================

            raw_results.append({

                "candidate_id": filename,

                "tfidf_score": float(
                    tfidf_score
                ),

                "good_matches": int(
                    good_match_count
                ),

                "ransac_inliers": int(
                    ransac_inliers
                ),

                "inlier_ratio": float(
                    inlier_ratio
                )
            })

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                candidate_number + 1
            ) % 100 == 0:

                print(
                    f"      "
                    f"{candidate_number + 1}/"
                    f"{len(candidates)} candidates"
                )

        # ====================================================
        # TIMING
        # ====================================================

        stage2_time = (
            time.time() - stage2_start
        )

        total_time = (
            time.time() - total_start
        )

        # ====================================================
        # RETURN RAW DATA
        # ====================================================

        return {

            "query_id": query_filename,

            "candidates": raw_results,

            "num_candidates": len(
                raw_results
            ),

            "stage1_time": float(
                stage1_time
            ),

            "stage2_time": float(
                stage2_time
            ),

            "total_time": float(
                total_time
            )
        }


# ============================================================
# QUERY PATH
# ============================================================

def get_query_path(query_index):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Change this path if your dataset folder has a different
    # location.
    #
    # Example expected:
    #
    # ../dataset/ukbench00000.jpg
    # ../dataset/ukbench00001.jpg
    # ...
    # --------------------------------------------------------

    return (
        f"../dataset/ukbench/"
        f"ukbench{query_index:05d}.jpg"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAW UKBENCH RETRIEVAL")
    print("=" * 70)

    print(
        f"Queries: {len(QUERY_INDICES)}"
    )

    print(
        f"Candidates per query: "
        f"{CANDIDATE_COUNT}"
    )

    print(
        f"Ratio threshold: "
        f"{RATIO_THRESHOLD}"
    )

    print(
        f"RANSAC threshold: "
        f"{RANSAC_THRESHOLD}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Initialize engine
    # --------------------------------------------------------

    engine = VisualSearchEngine()

    # --------------------------------------------------------
    # Store all query results
    # --------------------------------------------------------

    all_results = {}

    overall_start = time.time()

    # ========================================================
    # PROCESS QUERIES
    # ========================================================

    for query_number, query_index in enumerate(
        QUERY_INDICES,
        start=1
    ):

        query_path = get_query_path(
            query_index
        )

        query_filename = os.path.basename(
            query_path
        )

        print()
        print("=" * 70)

        print(
            f"QUERY "
            f"{query_number}/"
            f"{len(QUERY_INDICES)}"
        )

        print(
            f"Image: {query_filename}"
        )

        print("=" * 70)

        query_start = time.time()

        try:

            result = engine.search_raw(
                query_path
            )

            all_results[
                query_filename
            ] = result

            query_time = (
                time.time() - query_start
            )

            print(
                f"Stage 1: "
                f"{result['stage1_time']:.2f}s"
            )

            print(
                f"Stage 2: "
                f"{result['stage2_time']:.2f}s"
            )

            print(
                f"Total:   "
                f"{result['total_time']:.2f}s"
            )

            print(
                f"Candidates saved: "
                f"{len(result['candidates'])}"
            )

            print(
                f"Query wall time: "
                f"{query_time:.2f}s"
            )

        except Exception as e:

            print(
                f"ERROR processing "
                f"{query_filename}: {e}"
            )

            raise

        # ----------------------------------------------------
        # Save incrementally
        #
        # This is important because the retrieval stage is
        # expensive. If the script is interrupted, you don't
        # lose everything.
        # ----------------------------------------------------

        output_path = Path(
            OUTPUT_PATH
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with gzip.open(
            output_path,
            "wb"
        ) as f:

            pickle.dump(
                all_results,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        print(
            f"Progress saved: "
            f"{output_path}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.time() - overall_start
    )

    print()
    print("=" * 70)
    print("RAW RETRIEVAL COMPLETE")
    print("=" * 70)

    print(
        f"Queries processed: "
        f"{len(all_results)}"
    )

    print(
        f"Total time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()