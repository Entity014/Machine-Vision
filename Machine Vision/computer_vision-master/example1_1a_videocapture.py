import cv2

print(cv2.__version__)

cap = cv2.VideoCapture(0)

# CAP_SIZE = (1280,720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

ret, im = cap.read()  # ใช้อ่านภาพทีละ frame

print(im.shape)  # (ความสูง, ความกว้าง, Layer สี) --> BGR
# print(type(im))
# print(im[0, 0]) # Uint8
# print(im[0, 0, 0])
# print(type(im[0, 0, 0]))

cv2.imshow("camera", im)
cv2.imshow("blue channel", im[:, :, 0])
cv2.imshow("green channel", im[:, :, 1])
cv2.imshow("red channel", im[:, :, 2])
cv2.waitKey()
cap.release()
