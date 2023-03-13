import numpy as np
import cv2 as cv

def resize_img(img, scale=0.15):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)


org_img = resize_img(cv.imread(r'/home/soucs/Python/data/Sunset_org1.jpg'))
edit_img = resize_img(cv.imread(r'/home/soucs/Python/data/Sunset_edit.jpg'))

cv.imshow('Edit',edit_img)
cv.imshow('Org',org_img)

difference = edit_img-org_img

cv.imshow('Diff',difference)
cv.waitKey(0)