#Download images from https://drive.google.com/file/d/1KqllafwQiJR-Ronos3N-AHNfnoBb8I7H/view?usp=sharing

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 : 5, 8
# 2 : 6, 3
# 3 : 2, 4
# 4 : 2, 4
# 5 : 1, 7
# 6 : 3, 5
# 7 : 4, 3
# 8 : 5, 5
# 9 : 2, 6
# 10 : 4, 2

ans = np.array([[5, 8], [6, 3], [2, 4], [2, 4], [1, 7], [3, 5], [4, 3], [5, 5], [2, 6], [4, 2]])

def yellow_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    background = np.clip(background, 1, 255)
    background_color = cv2.merge([background] * 3)
    norm_im = (img / background_color) * 255
    norm_im = np.clip(norm_im, 0, 255).astype(np.uint8)
    mask_y = cv2.inRange(norm_im, (20, 240, 240), (150, 255, 255))
    opening_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    closing_y = cv2.morphologyEx(opening_y, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    contours_y, hierarchy_yellow = cv2.findContours(closing_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return closing_y, contours_y

def blue_processing(img):    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    background = np.clip(background, 1, 255)
    background_color = cv2.merge([background] * 3)
    norm_im = (img / background_color) * 190
    norm_im = np.clip(norm_im, 0, 255).astype(np.uint8)
    erode_img1 = cv2.erode(norm_im, np.ones((22,22), np.uint8))
    dilated_mask_b = cv2.dilate(erode_img1, np.ones((5,1), np.uint8))
    mask_b = cv2.inRange(dilated_mask_b,(180,145,0),(255,255,150))
    opening_y = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    erode_img2 = cv2.erode(opening_y, np.flipud(np.eye(7, dtype=np.uint8)))
    contours_b, hierarchy_blue = cv2.findContours(erode_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return erode_img2, contours_b


def coinCounting(filename):
    im = cv2.imread(filename)
    # target_size = (int(im.shape[1]/2),int(im.shape[0]/2))
    target_size = (int(500),int(500))
    im = cv2.resize(im,target_size)
    mask_yellow, contours_yellow = yellow_processing(im)
    mask_blue, contours_blue = blue_processing(im)

    yellow = len(contours_yellow)
    blue = len(contours_blue)

    # print('Yellow = ',yellow)
    # print('Blue = ', blue)
    result_im = im.copy()
    cv2.putText(
        result_im,
        f"[{yellow} , {blue}]",
        (50, 100),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=3,
        thickness=3,
        color=(0, 0, 255),
    )


    cv2.imshow('Original Image',result_im)
    # cv2.imshow('Yellow Coin', mask_yellow)
    # cv2.imshow('Blue Coin', mask_blue)
    cv2.waitKey()

    return [yellow,blue]

for i in range(1,11):
    result = coinCounting('.\CoinCounting\coin'+str(i)+'.jpg')
    print(f"{i} : {result} ")
    # if (ans[i - 1] == result).all():
    #     print(f"{i} : correct <{result}>")
    # else:
    #     print(f"{i} : wrong <{ans[i - 1]}>")
