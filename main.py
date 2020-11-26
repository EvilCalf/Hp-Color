import cv2
import numpy as np


def stretch(img):
    '''
    图像拉伸函数
    '''
    maxi = float(img.max())
    mini = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255/(maxi-mini)*img[i, j]-(255*mini)/(maxi-mini))

    return img


def dobinaryzation(img):
    '''
    二值化处理函数
    '''
    maxi = float(img.max())
    mini = float(img.min())

    x = maxi-((maxi-mini)/2)
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


def locate_Hp(img, afterimg):
    '''
    定位试剂号
    '''
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找出最大的三个区域
    block = []
    for c in contours:
        # 找出轮廓的左上点和右下点，由此计算它的面积和长度比
        r = find_rectangle(c)
        a = (r[2]-r[0])*(r[3]-r[1])  # 面积
        s = (r[2]-r[0])*(r[3]-r[1])  # 长度比

        block.append([r, a, s])
    # 选出面积最大的3个区域
    block = sorted(block, key=lambda b: b[1])[-3:]

    return block


def find_license(img):
    '''
    预处理函数
    '''
    m = 400*img.shape[0]/img.shape[1]

    # 压缩图像
    img_out = cv2.resize(img, (400, int(m)), interpolation=cv2.INTER_CUBIC)

    # BGR转换为灰度图像
    img = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

    # 灰度拉伸
    img = stretch(img)
    cv2.imshow('afterimg', img)
    cv2.waitKey(0)

    img = cv2.medianBlur(img, 3)
    cv2.imshow('afterimg', img)
    cv2.waitKey(0)
    '''进行开运算，用来去除噪声'''

    r = 16
    h = w = r*2+1
    kernel = np.zeros((h, w), np.uint8)
    cv2.circle(kernel, (r, r), r, 1, -1)
    # # 开运算
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('afterimg', img)
    cv2.waitKey(0)

    # 图像二值化
    img = dobinaryzation(img)
    cv2.imshow('afterimg', img)
    cv2.waitKey(0)

    # canny边缘检测
    img = cv2.Canny(img, img.shape[0], img.shape[1])
    # cv2.imshow('afterimg',img)
    # cv2.waitKey(0)

    '''消除小的区域，保留大块的区域，从而定位试剂'''
    # 进行闭运算
    kernel = np.ones((5, 19), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('afterimg',img)
    cv2.waitKey(0)

    # 消除小区域，定位试剂位置
    rect = locate_Hp(img, img)

    return rect, img_out


def cut_Hp(afterimg, rect):
    '''
    图像分割函数
    '''
    # 转换为宽度和高度
    rect[2] = rect[2]-rect[0]
    rect[3] = rect[3]-rect[1]
    rect_copy = tuple(rect.copy())
    rect = [0, 0, 0, 0]
    # 创建掩膜
    mask = np.zeros(afterimg.shape[:2], np.uint8)
    # 创建背景模型  大小只能为13*5，行数只能为1，单通道浮点型
    bgdModel = np.zeros((1, 65), np.float64)
    # 创建前景模型
    fgdModel = np.zeros((1, 65), np.float64)
    # 分割图像
    cv2.grabCut(afterimg, mask, rect_copy, bgdModel,
                fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_show = afterimg*mask2[:, :, np.newaxis]

    return img_show


def deal_Hp(licenseimg):
    '''
    试剂图片二值化
    '''
    gray_img = cv2.cvtColor(licenseimg, cv2.COLOR_BGR2GRAY)

    # 均值滤波  去除噪声
    kernel = np.ones((3, 3), np.float32)/9
    gray_img = cv2.filter2D(gray_img, -1, kernel)

    # 二值化处理
    ret, thresh = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

    return thresh


def find_end(start, arg, black, white, width, black_max, white_max):
    end = start+1
    for m in range(start+1, width-1):
        if (black[m] if arg else white[m]) > (0.98*black_max if arg else 0.98*white_max):
            end = m
            break
    return end


if __name__ == '__main__':
    img = cv2.imread('2.jpg', cv2.IMREAD_COLOR)
    # 预处理图像
    block, afterimg = find_license(img)
    for i in range(len(block)):
        # 框出试剂号
        rect = block[i][0]
        cv2.rectangle(afterimg, (rect[0], rect[1]),
                      (rect[2], rect[3]), (0, 255, 0), 2)
        # 分割试剂与背景
        cutimg = cut_Hp(afterimg, rect)
        cv2.imshow('cutimg', cutimg)
        cv2.waitKey(0)
