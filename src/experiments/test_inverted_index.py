import cv2
import numpy as np
import pickle
import os

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"
INDEX_PATH = "../results/inverted_index_2000.pkl"

QUERY_PATH = "../dataset/ukbench/ukbench00000.jpg"


# -----------------------------
# Load data
# -----------------------------

print("Loading database features...")
with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

print("Loading vocabulary...")
vocabulary = np.load(VOCABULARY_PATH)

print("Loading inverted index...")
with open(INDEX_PATH, "rb") as f:
    inverted_index = pickle.load(f)


# -----------------------------
# Extract query SIFT
# -----------------------------

image = cv2.imread(QUERY_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

print("\nQuery:", os.path.basename(QUERY_PATH))
print("Query descriptors:", len(descriptors))


# -----------------------------
# Assign descriptors to words
# -----------------------------

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.match(descriptors, vocabulary)

query_words = set()

for match in matches:
    query_words.add(match.trainIdx)

print("Unique visual words:", len(query_words))


# -----------------------------
# Retrieve candidate images
# -----------------------------

candidates = set()

for word_id in query_words:
    candidates.update(inverted_index[word_id])

print("\nTotal database images:", len(features))
print("Candidate images:", len(candidates))

reduction = 100 * (1 - len(candidates) / len(features))

print("Candidate reduction:", round(reduction, 2), "%")