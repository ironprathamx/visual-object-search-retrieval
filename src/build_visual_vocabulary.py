import cv2
import numpy as np
import time

from paths import SIFT_SAMPLE_PATH, VOCABULARY_PATH, RESULTS_DIR, require_file

SAMPLE_PATH = SIFT_SAMPLE_PATH

NUM_WORDS = 4000


print("Loading SIFT sample...")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

descriptors = np.load(
    require_file(
        SAMPLE_PATH,
        "Generate it with: python src/sample_sift_descriptors.py",
    )
)

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