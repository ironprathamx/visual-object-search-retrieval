import cv2
import numpy as np
import pickle
import time
import os

from paths import (
    SIFT_FEATURES_PATH,
    VOCABULARY_PATH,
    TFIDF_INDEX_PATH,
    require_file,
)


FEATURES_PATH = SIFT_FEATURES_PATH
INDEX_PATH = TFIDF_INDEX_PATH

CANDIDATE_COUNT = 600


class VisualSearchEngine:

    def __init__(self):

        # ---------------------------------------------
        # Load precomputed data
        # ---------------------------------------------

        with open(
            require_file(
                FEATURES_PATH,
                "Generate it with: python src/precompute_features.py",
            ),
            "rb",
        ) as f:
            self.features = pickle.load(f)

        self.vocabulary = np.load(
            require_file(
                VOCABULARY_PATH,
                "Generate it with: python src/build_visual_vocabulary.py",
            )
        )

        with open(
            require_file(
                INDEX_PATH,
                "Generate it with: python src/build_tfidf_index.py",
            ),
            "rb",
        ) as f:
            index = pickle.load(f)

        self.tfidf_vectors = index["tfidf_vectors"]
        self.idf = index["idf"]

        # ---------------------------------------------
        # Initialize feature detector and matchers
        # ---------------------------------------------

        self.sift = cv2.SIFT_create()

        self.bf_vocab = cv2.BFMatcher(cv2.NORM_L2)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)


    def search(self, QUERY_PATH, TOP_K, INCLUDE_QUERY=False):

        # ---------------------------------------------
        # Load query image
        # ---------------------------------------------

        image = cv2.imread(QUERY_PATH)

        if image is None:
            raise ValueError(
                f"Could not read image: {QUERY_PATH}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        total_start = time.time()

        # ---------------------------------------------
        # SIFT feature extraction
        # ---------------------------------------------

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
            QUERY_PATH
        )

        # ---------------------------------------------
        # Assign query descriptors to visual words
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Build query TF-IDF vector
        # ---------------------------------------------

        query_vector = np.zeros(
            len(self.vocabulary),
            dtype=np.float32
        )

        total_words = sum(
            word_counts.values()
        )

        for word_id, count in word_counts.items():

            tf = count / total_words

            query_vector[word_id] = (
                tf * self.idf[word_id]
            )

        norm = np.linalg.norm(
            query_vector
        )

        if norm > 0:
            query_vector /= norm

        # ---------------------------------------------
        # Stage 1: TF-IDF candidate retrieval
        # ---------------------------------------------

        scores = []

        for filename, vector in self.tfidf_vectors.items():

            if not INCLUDE_QUERY and filename == query_filename:
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

        candidates = scores[:CANDIDATE_COUNT]

        stage1_time = (
            time.time() - total_start
        )

        # ---------------------------------------------
        # Stage 2: SIFT matching + RANSAC
        # ---------------------------------------------

        verified_results = []

        stage2_start = time.time()

        for filename, tfidf_score in candidates:

            data = self.features[filename]

            database_descriptors = (
                data["descriptors"]
            )

            database_keypoints = (
                data["keypoints"]
            )

            if database_descriptors is None:
                continue

            # KNN matching
            knn_matches = self.bf.knnMatch(
                query_descriptors,
                database_descriptors,
                k=2
            )

            # Lowe ratio test
            good_matches = []

            for m, n in knn_matches:

                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            # -----------------------------------------
            # RANSAC
            # -----------------------------------------

            inliers = 0

            if len(good_matches) >= 4:

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

            verified_results.append(
                (
                    filename,
                    inliers,
                    tfidf_score,
                    len(good_matches)
                )
            )

        stage2_time = (
            time.time() - stage2_start
        )

        # ---------------------------------------------
        # Final ranking
        # ---------------------------------------------

# ---------------------------------------------
# Final ranking score
# ---------------------------------------------

        # Extract raw scores
        tfidf_scores = np.array(
            [x[2] for x in verified_results],
            dtype=np.float32
        )

        good_scores = np.array(
            [x[3] for x in verified_results],
            dtype=np.float32
        )

        inlier_scores = np.array(
            [x[1] for x in verified_results],
            dtype=np.float32
        )


        # ---------------------------------------------
        # Min-max normalization
        # ---------------------------------------------

        def min_max_normalize(values):

            minimum = values.min()
            maximum = values.max()

            if maximum == minimum:
                return np.zeros_like(values)

            return (
                (values - minimum)
                / (maximum - minimum)
            )


        tfidf_norm = min_max_normalize(
            tfidf_scores
        )

        good_norm = min_max_normalize(
            good_scores
        )

        inlier_norm = min_max_normalize(
            inlier_scores
        )


        # ---------------------------------------------
        # Weighted final score
        # ---------------------------------------------

        final_scores = (
            0.25 * tfidf_norm
            + 0.7 * np.log1p(good_norm)
            + 0.05 * np.log1p(inlier_norm)
        )


        # ---------------------------------------------
        # Attach final scores
        # ---------------------------------------------

        updated_results = []

        for i, result in enumerate(verified_results):

            filename = result[0]
            inliers = result[1]
            tfidf_score = result[2]
            good_matches = result[3]

            updated_results.append(
                (
                    filename,
                    inliers,
                    tfidf_score,
                    good_matches,
                    float(final_scores[i])
                )
            )


        verified_results = updated_results


        # ---------------------------------------------
        # Final ranking
        # ---------------------------------------------

        verified_results.sort(
            key=lambda x: x[4],
            reverse=True
        )
        results = []

        for rank, (
            filename,
            inliers,
            tfidf_score,
            good_matches,
            score
        ) in enumerate(
            verified_results[:TOP_K],
            start=1
        ):

            result = {
                "rank": rank,
                "image": filename,
                "inliers": inliers,
                "tfidf_score": tfidf_score,
                "good_matches": good_matches,
                "final_score": score
            }

            results.append(result)

        total_time = (
            time.time() - total_start
        )

        return {
            "results": results,
            "total_time": total_time,
            "stage1_time": stage1_time,
            "stage2_time": stage2_time
        }
