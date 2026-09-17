import cv2
import numpy as np

# Load images
image1 = cv2.imread("../dataset/ukbench/ukbench00117.jpg")
image2 = cv2.imread("../dataset/ukbench/ukbench00116.jpg")

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create SIFT
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create BFMatcher
bf = cv2.BFMatcher()

# KNN matching
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Lowe ratio test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))


# --------------------------------
# RANSAC + Homography
# --------------------------------
if len(good_matches) < 4:
    print("Not enough matches for homography.")

else:
    src_points = np.float32([
    keypoints1[m.queryIdx].pt
    for m in good_matches
    ])

    dst_points = np.float32([
    keypoints2[m.trainIdx].pt
    for m in good_matches
    ])

    H, mask = cv2.findHomography(
    src_points,
    dst_points,
    cv2.RANSAC,
    5.0
    )
    if H is None:
        print("Homography could not be estimated.")
    else:
        # Count inliers
        inlier_count = np.sum(mask)
        print(H)
        print("Verified matches:", inlier_count)
        inlier_matches = [
            m for m, inlier in zip(good_matches, mask.ravel())
            if inlier
        ]

        print("Verified matches:", len(inlier_matches))

        output = cv2.drawMatches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            inlier_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imshow(
            f"RANSAC Inliers: {len(inlier_matches)}",
            output
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()