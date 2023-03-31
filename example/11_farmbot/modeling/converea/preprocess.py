import cv2
import numpy as np
from rembg import remove
import glob
dirs = ['1','2','3']
for dir in dirs:
    paths = glob.glob('./{}/*.jpg'.format(dir))
    cnt = 0
    for path in paths:
        cnt+=1
        if cnt%3==0:
            continue
        path = str(path)
        img = cv2.imread(path)
        # print(img.shape)
        img = img[60: 420,:420]
        # print(img.shape)
        img = remove(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2)
        # print(img.shape)
        spath = './resample/' + path[2:]
        cv2.imwrite(spath, img)
# img = remove(img)


img = cv2.imread('./aiotimg.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# lower_white = np.array([75, 235, 235])
# upper_white = np.array([255, 255, 255])
#
# lower_green = np.array([0, 100, 50])
# upper_green = np.array([90, 255, 255])
#
# mask_green = cv2.inRange(hsv, lower_green, upper_green)
# # mask_white = cv2.inRange(img, lower_white, upper_white)
# #
# # mask = mask_white + mask_green
# maskrb = remove(mask_green)
#
# result = cv2.bitwise_and(img, img, mask = mask_green)
#
# tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# im_color = cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)

# cv2.imshow('origin', img)
# cv2.imshow('gray', gray)
# cv2.imshow('recolor', im_color)
# cv2.imshow('maskrb', maskrb)
# cv2.imshow('result', result)


# 사이즈
# rsimg = cv2.resize(img, (640,320))
# print(rsimg.shape)
# cv2.imshow('rsimg', rsimg)
#
#
# # 회전
# rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# rows, cols, channels = img.shape
# cv2.imshow('rotate_90', rotate_90)
#
# M = cv2.getRotationMatrix2D((cols/2,rows/2),-30,1)
# rotate_30 = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow('rotate_30', rotate_30)
#
# # 이동
# M = np.float32([[1,0,100],[0,1,50]])
# dst = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow('dst', dst)
#
# #랜덤 적용
# import random
# rotate = random.randint(0,360)
# x_shift = random.randint(-50,50)
# y_shift = random.randint(-50,50)
# print(rotate, x_shift, y_shift)
# rm = cv2.getRotationMatrix2D((cols/2,rows/2),rotate,1)
# sm = np.float32([[1,0,x_shift],[0,1,y_shift]])
# dst = cv2.warpAffine(img,rm,(cols,rows))
# dst = cv2.warpAffine(dst,sm,(cols,rows))
# cv2.imshow('rdst', dst)
#
#
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()