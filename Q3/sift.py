import numpy as np
import cv2 as cv
from transformed_imgs import *

# Create SIFT object
sift = cv.SIFT_create()

# Original Image SIFT
kp_org = sift.detect(gray(org_img), None)
org_sift = cv.drawKeypoints(org_img, kp_org, 0, (0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

# Scaled Image SIFT
kp_scaled = sift.detect(gray(scaled_img), None)
scaled_sift=cv.drawKeypoints(scaled_img, kp_scaled, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

# Rotated Image SIFT
kp_rotate = sift.detect(gray(rotate_img), None)
rotate_sift = cv.drawKeypoints(rotate_img, kp_rotate, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

# Affine Image SIFT
kp_affine = sift.detect(gray(affine_img), None)
affine_sift = cv.drawKeypoints(affine_img, kp_affine, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

# Perspective Image SIFT
kp_perspective = sift.detect(gray(perspective_img), None)
perspective_sift = cv.drawKeypoints(perspective_img, kp_perspective, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

# # View SIFT Images
# cv.imshow('Org Sift Image',org_sift)
# cv.imshow('Scaled Sift Image', scaled_sift)
# cv.imshow('Rotate Sift Image', cv.rotate(rotate_sift, cv.ROTATE_90_COUNTERCLOCKWISE))
# # cv.imshow('Rotate Sift Image', rotate_sift)
# cv.imshow('Affine Sift Image', affine_sift)
# cv.imshow('Perspective Sift Image', perspective_sift)


# Matching descriptors
# Using Brute-Force matcher
def match(trans_img, trans_kp, trans_name='', org_img=org_img, org_kp=kp_org):
    global sift
    org_kp, org_des = sift.compute(org_img,org_kp)
    trans_kp, trans_des = sift.compute(trans_img, trans_kp)
    matcher = cv.BFMatcher() # Create BF matcher object
    matches = matcher.match(org_des,trans_des)
    
    matched_img = cv.drawMatches(org_img, org_kp, trans_img, trans_kp, matches[:10], None)
    cv.imshow(f"{trans_name} Matches", matched_img)

# Creating dict of all transformations and calling match function
transformations = {'Scaled':(scaled_img,kp_scaled), 'Rotate':(rotate_img,kp_rotate), 
              'Affine':(affine_img,kp_affine), 'Perspective':(perspective_img,kp_perspective)}

# # Matching and viewing
# for t in transformations:
#     match(*transformations[t],trans_name=t)

cv.waitKey(0)
cv.destroyAllWindows()