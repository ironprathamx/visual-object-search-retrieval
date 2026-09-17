import cv2
import numpy as np
import pickle
import os

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"
INDEX_PATH = "../results/tfidf_index_2000.pkl"
DATASET_PATH = "../dataset/ukbench"

QUERY_NUMBER = 9
CANDIDATE_COUNT = 574

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_relevant_images(query_filename):
    query_number = int(
        query_filename.replace("ukbench", "").replace(".jpg", "")
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


def extract_query_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def build_query_tfidf(descriptors, vocabulary, idf, bf_vocab):

    matches = bf_vocab.match(descriptors, vocabulary)

    word_counts = {}

    for match in matches:
        word_id = match.trainIdx
        word_counts[word_id] = word_counts.get(word_id, 0) + 1

    vector = np.zeros(len(vocabulary), dtype=np.float32)

    total_descriptors = len(descriptors)

    for word_id, count in word_counts.items():
        tf = count / total_descriptors
        vector[word_id] = tf * idf[word_id]

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector


def match_and_verify(query_keypoints, query_descriptors,
                     database_keypoints, database_descriptors, bf):

    matches = bf.knnMatch(
        query_descriptors,
        database_descriptors,
        k=2
    )

    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return len(good_matches), 0

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

    if H is None or mask is None:
        return len(good_matches), 0

    inliers = int(np.sum(mask))

    return len(good_matches), inliers


# --------------------------------------------------
# Load everything
# --------------------------------------------------

print("Loading database...")
with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

print("Loading vocabulary...")
vocabulary = np.load(VOCABULARY_PATH)

print("Loading TF-IDF index...")
with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)

tfidf_vectors = index["tfidf_vectors"]
idf = index["idf"]

bf_vocab = cv2.BFMatcher(cv2.NORM_L2)
bf = cv2.BFMatcher(cv2.NORM_L2)

# --------------------------------------------------
# Query
# --------------------------------------------------

query_filename = f"ukbench{QUERY_NUMBER:05d}.jpg"
query_path = os.path.join(DATASET_PATH, query_filename)

print()
print("Query:", query_filename)

query_keypoints, query_descriptors = extract_query_features(query_path)

print("Query descriptors:", len(query_descriptors))

# --------------------------------------------------
# TF-IDF stage
# --------------------------------------------------

query_vector = build_query_tfidf(
    query_descriptors,
    vocabulary,
    idf,
    bf_vocab
)

scores = []

for filename, vector in tfidf_vectors.items():

    if filename == query_filename:
        continue

    score = float(np.dot(vector, query_vector))

    scores.append((filename, score))

scores.sort(key=lambda x: x[1], reverse=True)
ranked = scores
candidates = ranked[:CANDIDATE_COUNT]

# TF-IDF ranks
tfidf_rank = {
    filename: rank + 1
    for rank, (filename, score) in enumerate(ranked)
}

tfidf_score = {
    filename: score
    for filename, score in ranked
}

# --------------------------------------------------
# RANSAC stage
# --------------------------------------------------

results = []

for filename, score in candidates:

    data = features[filename]

    good_matches, inliers = match_and_verify(
        query_keypoints,
        query_descriptors,
        data["keypoints"],
        data["descriptors"],
        bf
    )

    results.append(
        (
            filename,
            inliers,
            good_matches,
            score
        )
    )

results.sort(key=lambda x: x[1], reverse=True)

# --------------------------------------------------
# Final ranks
# --------------------------------------------------

final_rank = {
    filename: rank + 1
    for rank, (filename, _, _, _) in enumerate(results)
}

# --------------------------------------------------
# Print relevant images
# --------------------------------------------------

relevant = get_relevant_images(query_filename)

print()
print("Relevant images:")
print("----------------")

for filename in sorted(relevant):

    result = next(
        (
            r for r in results
            if r[0] == filename
        ),
        None
    )

    print()
    print(filename)

    print(
        "  TF-IDF rank:",
        tfidf_rank.get(filename, "NOT IN CANDIDATES")
    )

    print(
        "  TF-IDF score:",
        f"{tfidf_score.get(filename, 0):.4f}"
    )

    if result:

        _, inliers, good_matches, score = result

        print(
            "  Good matches:",
            good_matches
        )

        print(
            "  RANSAC inliers:",
            inliers
        )

        print(
            "  Final rank:",
            final_rank[filename]
        )

    else:
        print("  Not evaluated by RANSAC.")

# --------------------------------------------------
# Top 15 final results
# --------------------------------------------------

print()
print("Top 15 final results:")
print("---------------------")

for i, (filename, inliers, good_matches, score) in enumerate(
    results[:15],
    start=1
):

    marker = " <-- RELEVANT" if filename in relevant else ""

    print(
        f"{i:2d}. {filename} | "
        f"inliers={inliers:3d} | "
        f"good={good_matches:3d} | "
        f"tfidf={score:.4f}"
        f"{marker}"
    )