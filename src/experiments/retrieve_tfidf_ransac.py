import cv2
import numpy as np
import pickle
import time
import os

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"
INDEX_PATH = "../results/tfidf_index_2000.pkl"

QUERY_PATH = "../dataset/ukbench/ukbench00000.jpg"

CANDIDATE_COUNT = 574
TOP_K = 10


# --------------------------------------------------
# Load database
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
# Extract query SIFT
# --------------------------------------------------

image = cv2.imread(QUERY_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

total_start = time.time()

query_keypoints, query_descriptors = sift.detectAndCompute(
    gray,
    None
)

query_filename = os.path.basename(QUERY_PATH)

print("\nQuery:", query_filename)
print("Query descriptors:", len(query_descriptors))


# --------------------------------------------------
# Assign query descriptors to visual words
# --------------------------------------------------

word_start = time.time()

bf_vocab = cv2.BFMatcher(cv2.NORM_L2)

matches = bf_vocab.match(
    query_descriptors,
    vocabulary
)

word_counts = {}

for match in matches:
    word_id = match.trainIdx
    word_counts[word_id] = word_counts.get(word_id, 0) + 1

print("Unique visual words:", len(word_counts))


# --------------------------------------------------
# Build query TF-IDF vector
# --------------------------------------------------

query_vector = np.zeros(
    len(vocabulary),
    dtype=np.float32
)

total_words = sum(word_counts.values())

for word_id, count in word_counts.items():

    tf = count / total_words

    query_vector[word_id] = tf * idf[word_id]


norm = np.linalg.norm(query_vector)

if norm > 0:
    query_vector /= norm


# --------------------------------------------------
# Stage 1: TF-IDF candidate retrieval
# --------------------------------------------------

scores = []

for filename, vector in tfidf_vectors.items():

    if filename == query_filename:
        continue

    similarity = np.dot(
        query_vector,
        vector
    )

    scores.append(
        (filename, float(similarity))
    )


scores.sort(
    key=lambda x: x[1],
    reverse=True
)

candidates = scores[:CANDIDATE_COUNT]

# Determine the correct UKBench group
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

print("\nRelevant image TF-IDF ranks:")

for target in sorted(relevant):
    for rank, (filename, score) in enumerate(scores, start=1):
        if filename == target:
            print(f"{target}: rank {rank}, TF-IDF={score:.4f}")
            break
word_end = time.time()

print(
    "\nStage 1 TF-IDF time:",
    word_end - word_start,
    "seconds"
)

print(
    "Candidates:",
    len(candidates)
)


# --------------------------------------------------
# Stage 2: SIFT matching + RANSAC
# --------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_L2)

verified_results = []

stage2_start = time.time()

for rank, (filename, tfidf_score) in enumerate(
    candidates,
    start=1
):

    data = features[filename]

    database_descriptors = data["descriptors"]
    database_keypoints = data["keypoints"]

    if (
        query_descriptors is None
        or database_descriptors is None
    ):
        continue

    # KNN matching
    knn_matches = bf.knnMatch(
        query_descriptors,
        database_descriptors,
        k=2
    )

    # Lowe ratio test
    good_matches = []

    for m, n in knn_matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # RANSAC
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

        if H is not None and mask is not None:
            inliers = int(np.sum(mask))

    verified_results.append(
        (
            filename,
            inliers,
            tfidf_score,
            len(good_matches)
        )
    )

stage2_end = time.time()


# --------------------------------------------------
# Final ranking
# --------------------------------------------------

verified_results.sort(
    key=lambda x: x[1],
    reverse=True
)

total_end = time.time()


# --------------------------------------------------
# Print results
# --------------------------------------------------

print(
    "\nStage 2 SIFT + RANSAC time:",
    stage2_end - stage2_start,
    "seconds"
)

print(
    "Total retrieval time:",
    total_end - total_start,
    "seconds"
)

print(
    "\nTop",
    TOP_K,
    "final results:"
)

for rank, (
    filename,
    inliers,
    tfidf_score,
    good_matches
) in enumerate(
    verified_results[:TOP_K],
    start=1
):

    print(
        f"{rank:2d}. {filename} "
        f"(inliers={inliers}, "
        f"good_matches={good_matches}, "
        f"tfidf={tfidf_score:.4f})"
    )


# --------------------------------------------------
# Recall@10
# --------------------------------------------------

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

top10 = [
    filename
    for filename, _, _, _
    in verified_results[:TOP_K]
]

correct = len(
    set(top10) & relevant
)

recall_at_10 = correct / 3

print("\nRelevant images:")

for filename in sorted(relevant):
    print(filename)

print(
    "\nCorrect in Top-10:",
    correct,
    "/ 3"
)

print(
    "Recall@10:",
    recall_at_10
)