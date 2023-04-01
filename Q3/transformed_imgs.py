import numpy as np
import cv2 as cv

# Function to compress big image
def resize_img(img, scale=0.25):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)
# Turn to Grayscale
def gray(img):
    return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

org_img = resize_img(cv.imread(r'/home/soucs/Python/computer-vision-hands-on/Q2/dataset/Sunset_org.jpg'))
dim = org_img.shape

# Scaled (down) and rotated (90deg clockwise) images
scaled_img = resize_img(org_img, scale=0.75)
rotate_img = cv.rotate(org_img, cv.ROTATE_90_CLOCKWISE)

# Affine Transformation image
srcTri = np.array( [[0, 0], [dim[1] - 1, 0], [0, dim[0] - 1]] ).astype(np.float32)
dstTri = np.array( [[0, dim[1]*0.33], [dim[1]*0.85, dim[0]*0.25], [dim[1]*0.15, dim[0]*0.7]]).astype(np.float32)
warp_mat = cv.getAffineTransform(srcTri, dstTri)
affine_img = cv.warpAffine(org_img, warp_mat, (dim[1], dim[0]))

# Perspective Transformation image
srcTri = np.array([[0, 260], [640, 260],[0, 400], [640, 400]],dtype=np.float32)
dstTri = np.array([[0, 0], [640, 260],[0, 400], [640, 400]],dtype=np.float32)
warp_mat = cv.getPerspectiveTransform(srcTri, dstTri)
perspective_img = cv.warpPerspective(org_img, warp_mat, (dim[1], dim[0]))

# # View Images
# cv.imshow('Original Image',org_img)
# cv.imshow('Scaled Image',scaled_img)
# cv.imshow('Rotate Image',rotate_img)
# cv.imshow('Affine Image',affine_img)
# cv.imshow('Perspective Image',perspective_img)
# cv.waitKey(0)
# cv.destroyAllWindows()