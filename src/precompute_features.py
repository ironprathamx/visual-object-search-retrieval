import cv2
import numpy as np
import os
import pickle
import time

from paths import DATASET_DIR, RESULTS_DIR, require_dir

dataset_path = str(
    require_dir(
        DATASET_DIR,
        "Download UKBench and extract images to dataset/ukbench/",
    )
)

sift = cv2.SIFT_create()

features = {}

start_time = time.time()

for filename in os.listdir(dataset_path):

    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(dataset_path, filename)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    keypoint_locations = np.float32([
        kp.pt for kp in keypoints
    ])

    features[filename] = {
        "keypoints": keypoint_locations,
        "descriptors": descriptors
    }

    print(filename, ":", len(keypoints), "keypoints")


RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / "sift_features.pkl", "wb") as f:
    pickle.dump(features, f)

print("Features saved.")
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")