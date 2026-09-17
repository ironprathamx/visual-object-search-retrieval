import pickle
import numpy as np

features_path = "../results/sift_features.pkl"

with open(features_path, "rb") as f:
    features = pickle.load(f)

keypoint_counts = [
    len(data["keypoints"])
    for data in features.values()
]

print("Images:", len(keypoint_counts))
print("Minimum:", min(keypoint_counts))
print("Maximum:", max(keypoint_counts))
print("Average:", np.mean(keypoint_counts))

print("\nImages below:")
print("< 10:", sum(k < 10 for k in keypoint_counts))
print("< 20:", sum(k < 20 for k in keypoint_counts))
print("< 50:", sum(k < 50 for k in keypoint_counts))
print("< 100:", sum(k < 100 for k in keypoint_counts))