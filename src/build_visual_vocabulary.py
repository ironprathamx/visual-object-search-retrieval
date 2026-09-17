import cv2
import numpy as np
import time

SAMPLE_PATH = "../results/sift_sample.npy"
VOCABULARY_PATH = "../results/visual_vocabulary_2000.npy"

NUM_WORDS = 2000


print("Loading SIFT sample...")

descriptors = np.load(SAMPLE_PATH)

print("Sample shape:", descriptors.shape)

# K-means termination criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    20,
    0.1
)

start = time.time()

print(f"\nRunning K-means with {NUM_WORDS} visual words...")

compactness, labels, centers = cv2.kmeans(
    descriptors,
    NUM_WORDS,
    None,
    criteria,
    3,
    cv2.KMEANS_PP_CENTERS
)

print("\nK-means finished.")

print("Vocabulary shape:", centers.shape)
print("Compactness:", compactness)
print("Time:", time.time() - start, "seconds")

np.save(VOCABULARY_PATH, centers)

print("\nSaved vocabulary to:")
print(VOCABULARY_PATH)