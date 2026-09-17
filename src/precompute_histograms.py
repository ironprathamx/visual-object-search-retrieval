import cv2
import os
import pickle
import numpy as np
import time

DATASET_PATH = "../dataset/ukbench"
OUTPUT_PATH = "../results/color_histograms.pkl"


def compute_histogram(image_path):
    image = cv2.imread(image_path)

    # Convert BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2D histogram using Hue and Saturation
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    # Normalize so image size does not matter
    cv2.normalize(histogram, histogram)

    return histogram.flatten().astype(np.float32)


histograms = {}

start = time.time()

files = sorted(
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

print("Images found:", len(files))

for i, filename in enumerate(files):

    image_path = os.path.join(DATASET_PATH, filename)

    histogram = compute_histogram(image_path)

    histograms[filename] = histogram

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(files)}")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(histograms, f)

print("\nSaved:", OUTPUT_PATH)
print("Total time:", time.time() - start, "seconds")