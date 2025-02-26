import numpy as np
import cv2


cap = cv2.VideoCapture(0)

detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

# detector = cv2.ORB_create()
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

ref = cv2.imread("ka/sheet.jpeg", cv2.COLOR_BGR2GRAY)
ref2 = cv2.imread("ka/oreo.jpeg", cv2.COLOR_BGR2GRAY)
h, w, _ = ref.shape
ref = cv2.resize(ref, (int(w * 1.2), int(h * 1.2)))
ref2 = cv2.resize(ref2, (int(w * 1.2), int(h * 1.2)))

kp1, des1 = detector.detectAndCompute(ref, None)
kp3, des3 = detector.detectAndCompute(ref2, None)

while True:
    _, target = cap.read()

    kp2, des2 = detector.detectAndCompute(target, None)

    matches = matcher.knnMatch(des1, des2, k=2)
    matches2 = matcher.knnMatch(des3, des2, k=2)

    good = []
    good2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    for m, n in matches2:
        if m.distance < 0.7 * n.distance:
            good2.append(m)

    if len(good) > 12:
        # print(len(good))
        ref_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(ref_pts, target_pts, cv2.RANSAC, 20.0)
        matchesMask = mask.ravel().tolist()

        h, w, _ = ref.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)

        cv2.polylines(target, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

    if len(good2) > 12:
        ref_pts2 = np.float32([kp3[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
        target_pts2 = np.float32([kp2[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)

        M2, mask2 = cv2.findHomography(ref_pts2, target_pts2, cv2.RANSAC, 20.0)
        matchesMask = mask2.ravel().tolist()

        h, w, _ = ref.shape
        pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst2 = cv2.perspectiveTransform(pts2, M2)

        cv2.polylines(target, [np.int32(dst2)], True, (255, 0, 0), 3, cv2.LINE_AA)

    # result = target
    # result = cv2.drawMatches(ref, kp1, target, kp2, good, None, flags=2)

    cv2.imshow("result", target)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
