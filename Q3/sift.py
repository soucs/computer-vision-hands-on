import numpy as np
import cv2 as cv
from transformed_imgs import *

# Create SIFT object
sift = cv.SIFT_create()

# Original Image SIFT
kp_org = sift.detect(gray(org_img), None)
org_sift = cv.drawKeypoints(org_img, kp_org, 0, (0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Org Sift Image',org_sift)

# Scaled Image SIFT
kp_scaled = sift.detect(gray(scaled_img), None)
# kp_scaled, _ = sift.compute(scaled_img,kp_scaled)
scaled_sift=cv.drawKeypoints(scaled_img, kp_scaled, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Scaled Sift Image', scaled_sift)

# Rotated Image SIFT
kp_rotate = sift.detect(gray(rotate_img), None)
rotate_sift = cv.drawKeypoints(rotate_img, kp_rotate, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Rotate Sift Image', cv.rotate(rotate_sift, cv.ROTATE_90_COUNTERCLOCKWISE))
# cv.imshow('Rotate Sift Image', rotate_sift)

# Affine Image SIFT
kp_affine = sift.detect(gray(affine_img), None)
affine_sift = cv.drawKeypoints(affine_img, kp_affine, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Affine Sift Image', affine_sift)

# Perspective Image SIFT
kp_perspective = sift.detect(gray(perspective_img), None)
perspective_sift = cv.drawKeypoints(perspective_img, kp_perspective, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Perspective Sift Image', perspective_sift)

cv.waitKey(0)
cv.destroyAllWindows()