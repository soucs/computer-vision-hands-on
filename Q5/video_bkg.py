# Ref: https://www.youtube.com/watch?v=p9eNXa_8j-k

import cv2 as cv
from external.SelfiSegmentationModule import SelfiSegmentation

vid = cv.VideoCapture(0)
vid.set(3, 640) # width
vid.set(4, 480) # height
bg_img = cv.imread('/home/soucs/Python/computer-vision-hands-on/Q5/dataset/milky-way.jpg')
seg = SelfiSegmentation()

while True:
    _, video = vid.read()
    vid_rmbg = seg.removeBG(video, bg_img, threshold=0.8)
    cv.imshow("Video", video)
    if cv.waitKey(1)==ord('x'):
        break