import cv2
import numpy as np
import pandas as pd


def dobinaryzation(img):
    '''
    二值化处理函数
    '''
    kernel = np.ones((3, 3), np.float32)/9
    img = cv2.filter2D(img, -1, kernel)

    maxi = float(img.max())
    mini = float(img.min())

    x = maxi-((maxi-mini)/10)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret, thresh = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
    # 返回二值化后的黑白图像
    return thresh


def find_rectangle(contour):
    '''
    寻找矩形轮廓
    '''
    y, x = [], []

    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]


def locate(img):
    '''
    定位试纸
    '''
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    block = []
    for c in contours:
        r = find_rectangle(c)
        a = (r[2]-r[0])*(r[3]-r[1])  # 面积
        s = (r[2]-r[0])*(r[3]-r[1])  # 长度比

        block.append([r, a, s])

    block = sorted(block, key=lambda b: b[1])

    return block


img_raw = cv2.imread('color.jpg', cv2.IMREAD_COLOR)
h,w,c=img_raw.shape

cropped = img_raw[int(h/8):int(h*7/8), int(w/8):int(w*7/8)]
    
# cv2.imshow('img', cropped)
# cv2.waitKey(0)
img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)
img = cv2.medianBlur(img, 3)
# df=pd.DataFrame(img)
# df.to_csv('img.csv')
cv2.imshow('img', img)
cv2.waitKey(0)

# img = cv2.addWeighted(img, 1, img, 1, -200)

# cv2.imshow('img', img)
# cv2.waitKey(0)

img = dobinaryzation(img)
cv2.imshow('img', img)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('img', img)   
cv2.waitKey(0) 
img = cv2.Canny(img, img.shape[0], img.shape[1])
cv2.imshow('img', img)   
cv2.waitKey(0) 


block = locate(img)

for i in range(len(block)):
    # 框出试剂号
    rect = block[i][0]
    cv2.rectangle(cropped, (rect[0], rect[1]),
                      (rect[2], rect[3]), (0, 255, 0), 1)
cv2.imshow('img', img_raw)

cv2.waitKey(0)
