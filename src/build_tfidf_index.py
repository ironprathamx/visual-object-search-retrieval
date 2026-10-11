import cv2
import numpy as np
import pickle
import time
import math

from paths import (
    SIFT_FEATURES_PATH,
    VOCABULARY_PATH,
    TFIDF_INDEX_PATH,
    RESULTS_DIR,
    require_file,
)

FEATURES_PATH = SIFT_FEATURES_PATH
INDEX_PATH = TFIDF_INDEX_PATH

print("Loading SIFT database...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(
    require_file(
        FEATURES_PATH,
        "Generate it with: python src/precompute_features.py",
    ),
    "rb",
) as f:
    features = pickle.load(f)

print("Database loaded.")

print("Loading visual vocabulary...")
vocabulary = np.load(
    require_file(
        VOCABULARY_PATH,
        "Generate it with: python src/build_visual_vocabulary.py",
    )
)

print("Vocabulary shape:", vocabulary.shape)

bf = cv2.BFMatcher(cv2.NORM_L2)

num_images = len(features)
num_words = len(vocabulary)

# --------------------------------------------------
# First pass:
# Assign every descriptor to a visual word
# and count word frequencies per image.
# --------------------------------------------------

image_word_counts = {}
document_frequency = np.zeros(num_words, dtype=np.int32)

start = time.time()

for i, (filename, data) in enumerate(features.items()):

    descriptors = data["descriptors"]

    if descriptors is None:
        continue

    matches = bf.match(descriptors, vocabulary)

    word_counts = {}

    for match in matches:
        word_id = match.trainIdx
        word_counts[word_id] = word_counts.get(word_id, 0) + 1

    image_word_counts[filename] = word_counts

    # Number of images containing each word
    for word_id in word_counts:
        document_frequency[word_id] += 1

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{num_images} images")


# --------------------------------------------------
# Calculate IDF
# --------------------------------------------------

print("\nCalculating IDF...")

idf = np.log(
    (num_images + 1) /
    (document_frequency + 1)
) + 1


# --------------------------------------------------
# Build TF-IDF vectors
# --------------------------------------------------

print("Building TF-IDF vectors...")

tfidf_vectors = {}

for filename, word_counts in image_word_counts.items():

    vector = np.zeros(num_words, dtype=np.float32)

    total_words = sum(word_counts.values())

    for word_id, count in word_counts.items():

        # Term frequency
        tf = count / total_words

        # TF-IDF
        vector[word_id] = tf * idf[word_id]

    # L2 normalization for cosine similarity
    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    tfidf_vectors[filename] = vector


# --------------------------------------------------
# Save
# --------------------------------------------------

index = {
    "tfidf_vectors": tfidf_vectors,
    "idf": idf,
    "document_frequency": document_frequency,
    "num_words": num_words
}

with open(INDEX_PATH, "wb") as f:
    pickle.dump(index, f)

print("\nFinished.")
print("Images:", len(tfidf_vectors))
print("Visual words:", num_words)
print("Time:", time.time() - start, "seconds")

print("\nSaved:")
print(INDEX_PATH)