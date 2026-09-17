import cv2
import os
import pickle
import numpy as np
import time

DATASET_PATH = "../dataset/ukbench"
OUTPUT_PATH = "../results/hog_features.pkl"


def compute_hog(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize all images to the same size
    gray = cv2.resize(gray, (128, 128))

    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude, angle = cv2.cartToPolar(
        gx,
        gy,
        angleInDegrees=True
    )

    # HOG uses unsigned orientations: 0-180 degrees
    angle = angle % 180

    # 16x16 cells
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


features = {}

start = time.time()

files = sorted(
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

print("Images found:", len(files))

for i, filename in enumerate(files):

    image_path = os.path.join(DATASET_PATH, filename)

    features[filename] = compute_hog(image_path)

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(files)}")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(features, f)

print("\nSaved:", OUTPUT_PATH)
print("Total time:", time.time() - start, "seconds")