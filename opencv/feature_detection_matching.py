import cv2 as cv
import numpy as np

# <----------harris corner detection----------->

# img = cv.imread('images/haikyuu.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# # apply harris corner detection
# dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.4)

# # dilate to mark the corner
# dst = cv.dilate(dst, None)

# # apply thresholds
# img[dst > 0.01 * dst.max()] = [0, 0, 255]

# cv.imshow('harris corner', img)
# cv.waitKey(0)

# <----------shi tomasi corner detection----------->

# img = cv.imread('images/giyu.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # parameters
# maxCorner = 100
# qualityLevel = 0.01
# minDistance = 10

# # deetct corners
# corners = cv.goodFeaturesToTrack(gray, maxCorner, qualityLevel, minDistance)
# corners = np.int8(corners)

# # draw corners
# for corner in corners:
#     x, y = corner.ravel()
#     cv.circle(img, (x, y), 5, (0, 255, 0), -1)

# cv.imshow('shi-tomasi corner', img)
# cv.waitKey(0)

# <-----------canny edge detector----------->

# img = cv.imread('images/giyu.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # apply gaussian blur
# blur = cv.GaussianBlur(gray, (5,5), 0)

# # canny edge detection
# edges = cv.Canny(blur, threshold1=50, threshold2=150)

# cv.imshow('original', gray)
# cv.imshow('edges', edges)
# cv.waitKey(0)

# <------------feature descriptors------------>
# <------------SIFT-------------->

# img = cv.imread("images/giyu.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # SIFT detector
# sift = cv.SIFT_create()

# # detect keypoint and compute descriptor
# keypoints, descriptors = sift.detectAndCompute(gray, None)

# # draw keypoint
# img_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# print(f"no.of keypoints :  {len(keypoints)}")
# print(f"descriptor SHAPE: {descriptors.shape}")

# cv.imshow('sift', img_keypoints)
# cv.waitKey(0)

# <----------ORB--------->

# img = cv.imread("images/giyu.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # orb object
# orb = cv.ORB_create(nfeatures=800)

# # detect keypoints and compute descriptors
# keypoints, descriptors = orb.detectAndCompute(gray, None)

# # draw keypoints
# img_keypoints = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

# print(f"no. of ORB keypoints: {len(keypoints)}")
# cv.imshow("orb keypoints", img_keypoints)
# cv.waitKey(0)

# <---------AKAZE---------->

# img = cv.imread("images/giyu.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# #akaze object
# akaze = cv.AKAZE_create()

# # detect keypoints and compute descriptor
# keypoints, descriptors = akaze.detectAndCompute(gray, None)

# # draw keypoints
# img_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv.imshow("akaze keypoints", img_keypoints)
# cv.waitKey(0)

# <--------feature matching----------->
# <--------brute force matcher----------->

# # load two images
# img1 = cv.imread("images/img1.jpg", cv.IMREAD_GRAYSCALE)
# img2 = cv.imread("images/img2.jpg", cv.IMREAD_GRAYSCALE)

# # orb object
# orb = cv.ORB_create(400)

# # detect and compute keypoint and descriptor
# ky1, desc1 = orb.detectAndCompute(img1, None)
# ky2, desc2 = orb.detectAndCompute(img2, None)

# # bf matcher 
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# # match descriptor
# matches = bf.match(desc1, desc2)

# # sort matches by distance
# matches = sorted(matches, key=lambda x: x.distance)

# # draw matches
# img_matches = cv.drawMatches(img1, ky1, img2, ky2, matches[:50], None, 
#                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# print(f"no. of matches: {len(matches)}")
# cv.imshow("bf matcher", img_matches)
# cv.waitKey(0) 

# <---------flann detector---------->

# img1 = cv.imread("images/img1.jpg", cv.IMREAD_GRAYSCALE)
# img2 = cv.imread("images/img2.jpg", cv.IMREAD_GRAYSCALE)

# # initialize sift for flann
# sift = cv.SIFT_create()
# ky1, desc1 = sift.detectAndCompute(img1, None)
# ky2, desc2 = sift.detectAndCompute(img2, None)

# # flann parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
# search_params = dict(checks=50)

# # create flann matcher
# flann = cv.FlannBasedMatcher(index_params, search_params)

# # find k=2 best matches for each descriptor
# matches = flann.knnMatch(desc1, desc2, k=2)

# # apply ratio test (lowe's ratio test)
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)

# print(f"Good matches: {len(good_matches)} / {len(matches)}")

# # draw matches
# img_matches = cv.drawMatches(img1, ky1, img2, ky2, good_matches, None, 
#                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # cv.imshow("flann macthing", img_matches)
# # cv.waitKey(0)

# # <----------finding homography------------>

# # extract location of good matches
# src_pts = np.float32([ky1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([ky2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# # find homography
# m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# # get dimension of first image
# h, w = img1.shape

# # define corner of first img
# pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# # transform corner to first image
# dst = cv.perspectiveTransform(pts, m)

# # draw bounding box in second img
# img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
# cv.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3)

# cv.imshow('object detection', img2_color)
# cv.waitKey(0)

# <---------blob detection----------->

img = cv.imread("images/blob.jpg")

# blob detector parameters
params = cv.SimpleBlobDetector_Params()

# filter by area
params.filterByArea = True
params.minArea = 100

# filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.5

# filter by inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# create detector
detector = cv.SimpleBlobDetector_create(params)

# detect blobs
keypoints = detector.detect(img)

# draw detected blobs
img_keypoints = cv.drawKeypoints(img, keypoints, None, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(f"no. of keypoints : {len(keypoints)}")
cv.imshow('blobs', img_keypoints)
cv.waitKey(0)