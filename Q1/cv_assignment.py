import numpy as np
import cv2 as cv

# Function to compress big image
def resize_img(img, scale=0.15):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)

# Reading Original and Edited images
org_img = resize_img(cv.imread(r'/home/soucs/Python/computer-vision-hands-on/Q1/dataset/Sunset_org.jpg'))
edit_img = resize_img(cv.imread(r'/home/soucs/Python/computer-vision-hands-on/Q1/dataset/Sunset_edit.jpg'))

# View Images
cv.imshow('Edit',edit_img)
cv.imshow('Org',org_img)

# Pixel difference image
difference = np.subtract(edit_img, org_img)

# Show difference
cv.imshow('Diff',difference)
cv.waitKey(0)