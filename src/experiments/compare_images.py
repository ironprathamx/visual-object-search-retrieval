import cv2
import numpy as np


def compare_images(image1_path, image2_path):

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return 0

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
        return 0

    verified_matches = np.sum(mask)

    return verified_matches

print(compare_images("../dataset/ukbench/ukbench00117.jpg","../dataset/ukbench/ukbench00116.jpg"))