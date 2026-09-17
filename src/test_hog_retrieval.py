import cv2
import pickle
import numpy as np
import os
import time

HOG_PATH = "../results/hog_features.pkl"
DATASET_PATH = "../dataset/ukbench"


def compute_hog(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (128, 128))

    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude, angle = cv2.cartToPolar(
        gx,
        gy,
        angleInDegrees=True
    )

    # Unsigned orientations: 0-180 degrees
    angle = angle % 180

    cell_size = 16
    bins = 9
    bin_width = 180 / bins

    features = []

    for y in range(0, 128, cell_size):
        for x in range(0, 128, cell_size):

            cell_magnitude = magnitude[
                y:y + cell_size,
                x:x + cell_size
            ]

            cell_angle = angle[
                y:y + cell_size,
                x:x + cell_size
            ]

            histogram = np.zeros(bins, dtype=np.float32)

            for i in range(cell_size):
                for j in range(cell_size):

                    bin_index = int(
                        cell_angle[i, j] / bin_width
                    )

                    if bin_index >= bins:
                        bin_index = bins - 1

                    histogram[bin_index] += cell_magnitude[i, j]

            features.extend(histogram)

    return np.array(features, dtype=np.float32)


def rank_images(query_path, top_k=10):

    print("Loading HOG features...")

    with open(HOG_PATH, "rb") as f:
        features = pickle.load(f)

    print("HOG features loaded.")

    query = compute_hog(query_path)

    results = []

    start = time.time()

    for filename, descriptor in features.items():

        if filename == os.path.basename(query_path):
            continue

        # Euclidean distance
        distance = np.linalg.norm(query - descriptor)

        results.append((filename, distance))

    # Smaller distance = more similar
    results.sort(key=lambda x: x[1])

    print("HOG ranking time:", time.time() - start)

    return results[:top_k]


if __name__ == "__main__":

    for query in ["ukbench01000.jpg", "ukbench00000.jpg"]:

        print("\nQuery:", query)

        query_path = os.path.join(DATASET_PATH, query)

        results = rank_images(query_path, top_k=10)

        print("Top results:")

        for filename, distance in results:
            print(f"{filename}: {distance:.4f}")