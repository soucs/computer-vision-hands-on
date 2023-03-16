# Ref: https://www.youtube.com/watch?v=p9eNXa_8j-k

import cv2 as cv
from external.SelfiSegmentationModule import SelfiSegmentation

WIDTH, HEIGHT = (640, 480)

def resize_img(img, width=WIDTH, height=HEIGHT ):
    dim = (640, 480)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)

# vid = cv.VideoCapture(0)
# vid.set(3, WIDTH) # width
# vid.set(4, HEIGHT) # height

# bg_img = resize_img(cv.imread('/home/soucs/Python/computer-vision-hands-on/Q5/dataset/milky-way.jpg'))
# seg = SelfiSegmentation()

# while True:
#     _, video = vid.read()
#     vid_rmbg = seg.removeBG(video, bg_img, threshold=0.85)
#     cv.imshow("Video", vid_rmbg)
#     if cv.waitKey(1)==ord('x'):
#         break

bg_img = resize_img(cv.imread('/home/soucs/Python/computer-vision-hands-on/Q5/dataset/milky-way.jpg'))
seg = SelfiSegmentation()

vid = cv.VideoCapture(r'/home/soucs/Python/computer-vision-hands-on/output.mp4')
while True:
    isTrue, frame = vid.read()
    frame = resize_img(frame)
    frame_rmbg = seg.removeBG(frame, bg_img, threshold=0.8)
    cv.imshow('Video', frame_rmbg)
    cv.waitKey(50)
    if cv.waitKey(1) == ord('x'):
        break

cv.destroyAllWindows()