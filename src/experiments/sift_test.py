import cv2
import matplotlib.pyplot as plt

# 1. Load image
image_path = "../dataset/ukbench/ukbench00000.jpg"
image = cv2.imread(image_path)

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Create SIFT detector
sift = cv2.SIFT_create()

# 4. Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 5. Draw keypoints
output = cv2.drawKeypoints(
    gray,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 6. Display
cv2.imshow(f"SIFT Keypoints: {len(keypoints)}",output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Print information
print("Number of keypoints:", len(keypoints))
print("Descriptor shape:", descriptors.shape)