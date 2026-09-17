import pickle
import numpy as np

INDEX_PATH = "../results/inverted_index_2000.pkl"

with open(INDEX_PATH, "rb") as f:
    inverted_index = pickle.load(f)

posting_sizes = []

for word_id, images in inverted_index.items():
    posting_sizes.append(len(images))

posting_sizes = np.array(posting_sizes)

print("Visual words:", len(posting_sizes))

print("\nPosting-list statistics:")
print("Minimum:", posting_sizes.min())
print("Maximum:", posting_sizes.max())
print("Mean:", posting_sizes.mean())
print("Median:", np.median(posting_sizes))

print("\nWords appearing in:")
print("<= 100 images:", np.sum(posting_sizes <= 100))
print("<= 500 images:", np.sum(posting_sizes <= 500))
print("<= 1000 images:", np.sum(posting_sizes <= 1000))
print("<= 2000 images:", np.sum(posting_sizes <= 2000))
print("<= 5000 images:", np.sum(posting_sizes <= 5000))
print("All 10200 images:", np.sum(posting_sizes == 10200))