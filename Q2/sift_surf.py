import numpy as np
import cv2 as cv

# Function to compress big image
def resize_img(img, scale=0.15):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)

org_img = resize_img(cv.imread(r'/home/soucs/Python/computer-vision-hands-on/Q2/dataset/Sunset_org.jpg'))
gray= cv.cvtColor(org_img,cv.COLOR_BGR2GRAY)
cv.imshow('Original Image',org_img)

# SIFT
# sift = cv.SIFT_create()
# kp = sift.detect(gray,None)
# print(kp)
# sift_img=cv.drawKeypoints(gray, kp, org_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imshow('Sift Image',sift_img)
# cv.waitKey(0)

# SURF
surf = cv.SURF(400)
kp, des = surf.detectAndCompute(org_img,None)
print(len(kp))
