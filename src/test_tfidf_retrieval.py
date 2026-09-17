import cv2
import numpy as np
import pickle
import time
import os

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"
INDEX_PATH = "../results/tfidf_index_2000.pkl"

QUERY_PATH = "../dataset/ukbench/ukbench05000.jpg"

TOP_N = 50


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading database features...")
with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

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

start = time.time()

keypoints, descriptors = sift.detectAndCompute(gray, None)

print("\nQuery:", os.path.basename(QUERY_PATH))
print("Query descriptors:", len(descriptors))


# --------------------------------------------------
# Assign query descriptors to visual words
# --------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.match(descriptors, vocabulary)

word_counts = {}

for match in matches:
    word_id = match.trainIdx
    word_counts[word_id] = word_counts.get(word_id, 0) + 1

print("Unique visual words:", len(word_counts))


# --------------------------------------------------
# Build query TF-IDF vector
# --------------------------------------------------

query_vector = np.zeros(len(vocabulary), dtype=np.float32)

total_words = sum(word_counts.values())

for word_id, count in word_counts.items():

    tf = count / total_words

    query_vector[word_id] = tf * idf[word_id]


# Normalize

norm = np.linalg.norm(query_vector)

if norm > 0:
    query_vector /= norm


# --------------------------------------------------
# Calculate cosine similarity
# --------------------------------------------------

scores = []

for filename, vector in tfidf_vectors.items():

    similarity = np.dot(query_vector, vector)

    scores.append((filename, float(similarity)))


# Sort highest similarity first

scores.sort(key=lambda x: x[1], reverse=True)
query_filename = os.path.basename(QUERY_PATH)

scores = [
    (filename, score)
    for filename, score in scores
    if filename != query_filename
]

elapsed = time.time() - start


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nTF-IDF retrieval time:", elapsed, "seconds")

print("\nTop", TOP_N, "results:")

for rank, (filename, score) in enumerate(scores[:TOP_N], start=1):

    print(
        f"{rank:2d}. {filename} "
        f"(similarity={score:.4f})"
    )


# --------------------------------------------------
# Recall@10
# --------------------------------------------------

query_number = int(
    os.path.basename(QUERY_PATH)
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

query_filename = os.path.basename(QUERY_PATH)

relevant.remove(query_filename)

top10 = [filename for filename, score in scores[:10]]

correct = len(set(top10) & relevant)

recall_at_10 = correct / 3

print("\nRelevant images:")
for filename in sorted(relevant):
    print(filename)

print("\nCorrect in Top-10:", correct, "/ 3")
print("Recall@10:", recall_at_10)