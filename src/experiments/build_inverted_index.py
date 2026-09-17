import cv2
import numpy as np
import pickle
import time
import os

FEATURES_PATH = "../results/sift_features.pkl"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"
INDEX_PATH = "../results/inverted_index_2000.pkl"


print("Loading SIFT database...")

with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

print("Database loaded.")

print("Loading visual vocabulary...")

vocabulary = np.load(VOCABULARY_PATH)

print("Vocabulary shape:", vocabulary.shape)


# BF matcher for assigning descriptors to visual words
bf = cv2.BFMatcher(cv2.NORM_L2)


# Inverted index:
# visual_word_id -> set of image filenames
inverted_index = {
    word_id: set()
    for word_id in range(len(vocabulary))
}


start = time.time()

total_descriptors = 0

for i, (filename, data) in enumerate(features.items()):

    descriptors = data["descriptors"]

    if descriptors is None:
        continue

    total_descriptors += len(descriptors)

    # Find nearest visual word for every descriptor
    matches = bf.match(
        descriptors,
        vocabulary
    )

    # Add image to the corresponding visual-word lists
    for match in matches:

        word_id = match.trainIdx

        inverted_index[word_id].add(filename)

    if (i + 1) % 500 == 0:
        print(
            f"Processed {i + 1}/10200 images | "
            f"Descriptors: {total_descriptors}"
        )


# Convert sets to lists so the structure is easy to pickle
inverted_index = {
    word_id: list(image_names)
    for word_id, image_names in inverted_index.items()
}


with open(INDEX_PATH, "wb") as f:
    pickle.dump(inverted_index, f)


print("\nFinished.")

print("Total descriptors processed:", total_descriptors)
print("Visual words:", len(inverted_index))
print("Time:", time.time() - start, "seconds")

print("\nSaved:")
print(INDEX_PATH)