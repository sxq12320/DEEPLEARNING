import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#对视频进行读取

vc = cv.VideoCapture('0.picture\VID_20231003_123113.mp4') #读取视频
#vc = cv.VideoCapture(0) #读取视频
print(vc)

#视频的读取
if vc.isOpened():
    open,frame = vc.read() #返回两个数值 bool  当前帧的图像
    print(1)
else :
    print(0)
    open = False

while(open):
    print(1)
    ret,frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        cv.imshow('gray',gray)
        #plt.show(gray)
        if( cv.waitKey(30) & 0XFF == ord('q')):
            break

vc.release()
cv.destroyAllWindows()