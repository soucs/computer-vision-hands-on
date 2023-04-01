import numpy as np
import cv2 as cv
from transformed_imgs import *

# Create ORB object
orb = cv.ORB_create()

# Original Image ORB
kp_org = orb.detect(gray(org_img),None)
# kp_org, org_des = orb.compute(org_img, kp_org)
org_orb = cv.drawKeypoints(org_img, kp_org, None, color=(0,255,0), flags=0)
cv.imshow('Orginal ORB',org_orb)

# Scaled Image ORB
kp_scaled = orb.detect(gray(scaled_img), None)
# kp_scaled, scaled_des = orb.compute(scaled_img, kp_scaled)
scaled_orb = cv.drawKeypoints(scaled_img, kp_scaled, 0, (0, 0, 255), flags=0)
cv.imshow('Scaled ORB Image', scaled_orb)

# Rotated Image SIFT
kp_rotate = orb.detect(gray(rotate_img), None)
rotate_orb = cv.drawKeypoints(rotate_img, kp_rotate, 0, (0, 0, 255), flags=0)
cv.imshow('Rotate ORB Image', cv.rotate(rotate_orb, cv.ROTATE_90_COUNTERCLOCKWISE))
# cv.imshow('Rotate ORB Image', rotate_orb)

# Affine Image SIFT
kp_affine = orb.detect(gray(affine_img), None)
affine_orb = cv.drawKeypoints(affine_img, kp_affine, 0, (0, 0, 255), flags=0)
cv.imshow('Affine ORB Image', affine_orb)

# Perspective Image SIFT
kp_perspective = orb.detect(gray(perspective_img), None)
perspective_orb = cv.drawKeypoints(perspective_img, kp_perspective, 0, (0, 0, 255), flags=0)
cv.imshow('Perspective ORB Image', perspective_orb)

cv.waitKey(0)
cv.destroyAllWindows()