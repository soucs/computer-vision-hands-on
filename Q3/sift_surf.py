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
# cv.imshow('Original Image',org_img)

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
# dstTri = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])
dstTri = np.array([[0, 0], [640, 260],[0, 400], [640, 400]],dtype=np.float32)
warp_mat = cv.getPerspectiveTransform(srcTri, dstTri)
perspective_img = cv.warpPerspective(org_img, warp_mat, (dim[1], dim[0]))

# cv.imshow('Transformed Image', perspective_img)

# Original SIFT
sift = cv.SIFT_create()
kp_org = sift.detect(gray(org_img), None)
org_sift = cv.drawKeypoints(org_img, kp_org, 0, (0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Org Sift Image',org_sift)

# Scaled SIFT
kp_scaled = sift.detect(gray(scaled_img), None)
scaled_sift=cv.drawKeypoints(scaled_img, kp_scaled, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Scaled Sift Image', scaled_sift)

# Rotated SIFT
kp_rotate = sift.detect(gray(rotate_img), None)
rotate_sift = cv.drawKeypoints(rotate_img, kp_rotate, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Rotate Sift Image', cv.rotate(rotate_sift, cv.ROTATE_90_COUNTERCLOCKWISE))
# cv.imshow('Rotate Sift Image', rotate_sift)

# Affine SIFT
kp_affine = sift.detect(gray(affine_img), None)
affine_sift = cv.drawKeypoints(affine_img, kp_affine, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Affine Sift Image', affine_sift)

# Perspective SIFT
kp_perspective = sift.detect(gray(perspective_img), None)
perspective_sift = cv.drawKeypoints(perspective_img, kp_perspective, 0, (0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow('Perspective Sift Image', perspective_sift)

cv.waitKey(0)



# # SURF
# surf = cv.SURF(400)
# kp, des = surf.detectAndCompute(org_img,None)
# print(len(kp))
