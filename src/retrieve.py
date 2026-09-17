import cv2
import numpy as np
import pickle
import time

FEATURES_PATH = "../results/sift_features.pkl"
DATASET_PATH = "../dataset/ukbench"

def extract_query_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    keypoint_locations = np.float32([
        kp.pt for kp in keypoints
    ])

    return keypoint_locations, descriptors

bf = cv2.BFMatcher()


FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=25
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)


def get_good_matches(
    query_descriptors,
    database_descriptors
):

    if query_descriptors is None or database_descriptors is None:
        return []


    matches = bf.knnMatch(
        query_descriptors,
        database_descriptors,
        k=2
    )

    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches
def geometric_verification(
    query_keypoints,
    database_keypoints,
    good_matches
):

    if len(good_matches) < 4:
        return 0

    src_points = np.float32([
        query_keypoints[m.queryIdx]
        for m in good_matches
    ])

    dst_points = np.float32([
        database_keypoints[m.trainIdx]
        for m in good_matches
    ])

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )

    if H is None or mask is None:
        return 0

    return int(np.sum(mask))


def retrieve(query_path, top_k=10):
    # Load precomputed database features
    print("Loading database features...")

    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    print("Database loaded.")

    # Extract SIFT features from query
    query_keypoints, query_descriptors = extract_query_features(
        query_path
    )
    stage1_start = time.time()
    results = []

    for filename, data in features.items():

        # Don't compare the image with itself
        if filename == query_path.split("/")[-1]:
            continue

        good_matches = get_good_matches(
            query_descriptors,
            data["descriptors"]
        )
        score = len(good_matches)
        results.append((filename, score, good_matches))

    # Highest score first
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )
    candidates = results[:50]
    stage1_end = time.time()
    print("Stage 1 time:", stage1_end - stage1_start, "seconds")

    stage2_start = time.time()
    verified_results = []
    for filename, appearance_score, good_matches in candidates:
        data = features[filename]

        inliers = geometric_verification(
            query_keypoints,
            data["keypoints"],
            good_matches
        )

        verified_results.append(
            (filename, inliers)
        )

    verified_results.sort(
        key=lambda x: x[1],
        reverse=True
    )
    stage2_end = time.time()
    print("Stage 2 time:", stage2_end - stage2_start, "seconds")
    return verified_results[:top_k]


if __name__ == "__main__":
    start_time = time.time()
    query_path = "../dataset/ukbench/ukbench00000.jpg"

    results = retrieve(
        query_path,
        top_k=10
    )

    print("\nTop results:")

    for filename, score in results:
        print(filename, ":", score, "inliers")

    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")