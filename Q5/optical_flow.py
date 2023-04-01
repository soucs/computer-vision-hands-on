import cv2 as cv
import numpy as np

# Video input
cap = cv.VideoCapture(r'/home/soucs/Python/computer-vision-hands-on/Q5/dataset/output.mp4')

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        print('Frame not Found!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    # Dense Flow Calculate
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # Converting flow video from hsv to bgr
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('Optical Flow', bgr)
    if cv.waitKey(30) == ord('x'):
        break
    prvs = next
cv.destroyAllWindows()