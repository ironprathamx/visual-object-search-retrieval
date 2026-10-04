import cv2
import numpy as np
import pickle
import time
import os


FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"
INDEX_PATH = "../results/tfidf_index_2000.pkl"

CANDIDATE_COUNT = 574


class VisualSearchEngine:

    def __init__(self):

        # ---------------------------------------------
        # Load precomputed data
        # ---------------------------------------------

        with open(FEATURES_PATH, "rb") as f:
            self.features = pickle.load(f)

        self.vocabulary = np.load(VOCABULARY_PATH)

        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)

        self.tfidf_vectors = index["tfidf_vectors"]
        self.idf = index["idf"]

        # ---------------------------------------------
        # Initialize feature detector and matchers
        # ---------------------------------------------

        self.sift = cv2.SIFT_create()

        self.bf_vocab = cv2.BFMatcher(cv2.NORM_L2)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)


    def search(self, QUERY_PATH, TOP_K):

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

                if m.distance < 0.75 * n.distance:
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

        verified_results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for rank, (
            filename,
            inliers,
            tfidf_score,
            good_matches
        ) in enumerate(
            verified_results[:TOP_K],
            start=1
        ):

            result = {
                "rank": rank,
                "image": filename,
                "inliers": inliers,
                "tfidf_score": tfidf_score,
                "good_matches": good_matches
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
