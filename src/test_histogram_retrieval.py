import cv2
import pickle
import numpy as np
import os
import time

HISTOGRAMS_PATH = "../results/color_histograms.pkl"
DATASET_PATH = "../dataset/ukbench"


def compute_histogram(image_path):
    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(histogram, histogram)

    return histogram.flatten().astype(np.float32)


def rank_images(query_path, top_k=10):

    with open(HISTOGRAMS_PATH, "rb") as f:
        histograms = pickle.load(f)

    query_histogram = compute_histogram(query_path)

    results = []

    start = time.time()

    for filename, histogram in histograms.items():

        if filename == os.path.basename(query_path):
            continue

        # Correlation: higher = more similar
        score = cv2.compareHist(
            query_histogram,
            histogram,
            cv2.HISTCMP_CORREL
        )

        results.append((filename, score))

    results.sort(key=lambda x: x[1], reverse=True)

    print("Histogram ranking time:", time.time() - start)

    return results[:top_k]


if __name__ == "__main__":

    for query in ["ukbench01000.jpg", "ukbench00000.jpg"]:

        print("\nQuery:", query)

        query_path = os.path.join(DATASET_PATH, query)

        results = rank_images(query_path, top_k=10)

        for filename, score in results:
            print(f"{filename}: {score:.4f}")