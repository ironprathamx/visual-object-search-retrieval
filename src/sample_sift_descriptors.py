import pickle
import numpy as np
import time

FEATURES_PATH = "../results/sift_features.pkl"
OUTPUT_PATH = "../results/sift_sample.npy"

SAMPLE_SIZE = 100_000

print("Loading database features...")

with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

print("Database loaded.")

start = time.time()

# First determine how many descriptors exist
total_descriptors = 0

for data in features.values():

    descriptors = data["descriptors"]

    if descriptors is not None:
        total_descriptors += len(descriptors)

print("Total descriptors:", total_descriptors)

# Random descriptor indices
rng = np.random.default_rng(42)

selected_indices = set(
    rng.choice(
        total_descriptors,
        size=SAMPLE_SIZE,
        replace=False
    )
)

sample = []

current_index = 0

for filename, data in features.items():

    descriptors = data["descriptors"]

    if descriptors is None:
        continue

    for descriptor in descriptors:

        if current_index in selected_indices:
            sample.append(descriptor)

        current_index += 1

        if len(sample) == SAMPLE_SIZE:
            break

    if len(sample) == SAMPLE_SIZE:
        break

sample = np.asarray(sample, dtype=np.float32)

print("Sample shape:", sample.shape)

np.save(OUTPUT_PATH, sample)

print("Saved:", OUTPUT_PATH)
print("Time:", time.time() - start, "seconds")