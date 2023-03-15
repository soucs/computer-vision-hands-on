import numpy as np
import cv2 as cv

# Function to compress big image
def resize_img(img, scale=0.25):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)

def gray(img):
    return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

org_img = resize_img(cv.imread(r'/home/soucs/Python/computer-vision-hands-on/Q2/dataset/Sunset_org.jpg'))
cv.imshow('Original Image',org_img)

scaled_img = resize_img(org_img, scale=0.75)
rotate_img = cv.rotate(org_img, cv.ROTATE_90_CLOCKWISE)

# cv.imshow('Rotated Image',rotate_img)

# SIFT
sift = cv.SIFT_create()
kp_org = sift.detect(gray(org_img), None)
org_sift=cv.drawKeypoints(org_img, kp_org, 0, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Org Sift Image',org_sift)

kp_scaled = sift.detect(gray(scaled_img), None)
scaled_sift=cv.drawKeypoints(scaled_img, kp_scaled, 0, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imshow('Scaled Sift Image', scaled_sift)

kp_rotate = sift.detect(gray(rotate_img), None)
rotate_sift=cv.drawKeypoints(rotate_img, kp_rotate, 0, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imshow('Rotate Sift Image', cv.rotate(rotate_sift, cv.ROTATE_90_COUNTERCLOCKWISE))


cv.waitKey(0)



# # SURF
# surf = cv.SURF(400)
# kp, des = surf.detectAndCompute(org_img,None)
# print(len(kp))
