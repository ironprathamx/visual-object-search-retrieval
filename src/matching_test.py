import cv2
import matplotlib.pyplot as plt

# Load images
image1 = cv2.imread("../dataset/ukbench/ukbench00000.jpg")
image2 = cv2.imread("../dataset/ukbench/ukbench00001.jpg")

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create SIFT
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))

# Create BFMatcher
bf = cv2.BFMatcher()

# KNN matching
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

print("Total KNN matches:", len(matches))

# Ratio test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))

# Draw matches
output = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

cv2.imshow(f"Good SIFT Matches: {len(good_matches)}",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
