import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
TODO : Classification (Pixel-wise classification)
    - Segmentation
        - SVM
        - NB
"""

#Download files from https://drive.google.com/file/d/1XdZLvORnCnfpyBYflh15I58VQrQdVlUe/view?usp=sharing

im = cv2.imread("SkinDetection\SkinTrain1.jpg") # BGR
mask = cv2.imread("SkinDetection\SkinTrain1_mask.jpg",0) # binary -> 0, 255

im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV) # HSV
h = im_hsv[:,:,0] # 0, 179
s = im_hsv[:,:,1] # 0, 255

h_skin = h[mask >= 128]
s_skin = s[mask >= 128]
h_nonskin = h[mask < 128]
s_nonskin = s[mask < 128]


cv2.imshow('image',im)
cv2.imshow('mask',mask)
cv2.imshow('hue',h)
cv2.imshow('saturation',s)

# X-axis : hue, Y-axis : saturation
# Red is skin, Blue is Non skin
plt.plot(h_nonskin,s_nonskin,'b.')
plt.plot(h_skin,s_skin,'r.')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
