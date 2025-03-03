import numpy as np
import cv2

cap = cv2.VideoCapture(0)

TARGET_SIZE = (640, 360)

while True:
    ret, im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)

    mask = cv2.inRange(
        im_flipped, (0, 0, 90), (50, 50, 255)
    )  # b: 0 -> 50, g: 0 -> 50, r: 90 -> 255
    cv2.imshow("mask", mask)  # only values is 0 or 255
    cv2.moveWindow("mask", TARGET_SIZE[0], 0)

    print(np.sum(mask / 255))

    if np.sum(mask / 255) > 0.1 * im_flipped.shape[0] * im_flipped.shape[1]:
        cv2.putText(
            im_flipped,
            "Coke",
            (50, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=5,
            thickness=3,
            color=(0, 0, 255),
        )

    cv2.imshow("camera", im_flipped)
    cv2.moveWindow("camera", 0, 0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
