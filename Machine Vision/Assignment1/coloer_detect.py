import numpy as np
import cv2

cap = cv2.VideoCapture(0)

TARGET_SIZE = (640, 360)

while True:
    ret, im = cap.read()
    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)

    mask_coke = cv2.inRange(
        im_flipped, (0, 0, 90), (50, 50, 255)
    )  # b: 0 -> 50, g: 0 -> 50, r: 90 -> 255
    mask_pepsi = cv2.inRange(im_flipped, (30, 0, 0), (80, 20, 20))
    mask_sprite = cv2.inRange(im_flipped, (0, 50, 0), (50, 120, 20))

    cv2.imshow("mask_coke", mask_coke)  # only values is 0 or 255
    cv2.imshow("mask_pepsi", mask_pepsi)  # only values is 0 or 255
    cv2.imshow("mask_sprite", mask_sprite)  # only values is 0 or 255
    cv2.moveWindow("mask_coke", TARGET_SIZE[0], 0)
    cv2.moveWindow("mask_sprite", 0, TARGET_SIZE[1])
    cv2.moveWindow("mask_pepsi", TARGET_SIZE[0], TARGET_SIZE[1])

    print(np.sum(mask_coke / 255))

    if np.sum(mask_coke / 255) > 0.05 * im_flipped.shape[0] * im_flipped.shape[1]:
        cv2.putText(
            im_flipped,
            "Coke",
            (50, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=5,
            thickness=3,
            color=(0, 0, 255),
        )
    elif np.sum(mask_pepsi / 255) > 0.05 * im_flipped.shape[0] * im_flipped.shape[1]:
        cv2.putText(
            im_flipped,
            "pepsi",
            (50, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=5,
            thickness=3,
            color=(255, 0, 0),
        )
    elif np.sum(mask_sprite / 255) > 0.05 * im_flipped.shape[0] * im_flipped.shape[1]:
        cv2.putText(
            im_flipped,
            "sprite",
            (50, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=5,
            thickness=3,
            color=(0, 255, 0),
        )
    cv2.imshow("camera", im_flipped)
    cv2.moveWindow("camera", 0, 0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
