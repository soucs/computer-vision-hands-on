import numpy as np
import cv2 as cv
from transformed_imgs import *

# Create ORB object
orb = cv.ORB_create()

# Original Image ORB
kp_org = orb.detect(gray(org_img),None)
org_orb = cv.drawKeypoints(org_img, kp_org, None, color=(0,255,0), flags=0)

# Scaled Image ORB
kp_scaled = orb.detect(gray(scaled_img), None)
scaled_orb = cv.drawKeypoints(scaled_img, kp_scaled, 0, (0, 0, 255), flags=0)

# Rotated Image SIFT
kp_rotate = orb.detect(gray(rotate_img), None)
rotate_orb = cv.drawKeypoints(rotate_img, kp_rotate, 0, (0, 0, 255), flags=0)

# Affine Image SIFT
kp_affine = orb.detect(gray(affine_img), None)
affine_orb = cv.drawKeypoints(affine_img, kp_affine, 0, (0, 0, 255), flags=0)

# Perspective Image SIFT
kp_perspective = orb.detect(gray(perspective_img), None)
perspective_orb = cv.drawKeypoints(perspective_img, kp_perspective, 0, (0, 0, 255), flags=0)

# # View ORB images
# cv.imshow('Orginal ORB',org_orb)
# cv.imshow('Scaled ORB Image', scaled_orb)
# cv.imshow('Rotate ORB Image', cv.rotate(rotate_orb, cv.ROTATE_90_COUNTERCLOCKWISE))
# # cv.imshow('Rotate ORB Image', rotate_orb)
# cv.imshow('Affine ORB Image', affine_orb)
# cv.imshow('Perspective ORB Image', perspective_orb)

# Matching descriptors
# Using Brute-Force matcher
def match(trans_img, trans_kp, trans_name='', org_img=org_img, org_kp=kp_org):
    global orb
    org_kp, org_des = orb.compute(org_img,org_kp)
    trans_kp, trans_des = orb.compute(trans_img, trans_kp)
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