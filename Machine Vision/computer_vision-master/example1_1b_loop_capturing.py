import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, im = cap.read()

    cv2.imshow("camera", im)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


"""
1.HSV
2.Filter Blur
3.Histogram
4.Morphological
5.HOG
6.Image to text
7.Probability KNN Decision Tree
8.SVM
9.Canny
10.LSI Median
11.Overflow
12.Overfitting
"""
