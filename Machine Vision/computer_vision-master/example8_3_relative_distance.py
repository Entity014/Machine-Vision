import numpy as np
import cv2

"""
? โค้ดที่ให้มานี้เป็นสคริปต์ Python ที่ใช้ OpenCV เพื่อทำการจับคู่คุณลักษณะ (feature matching) แบบเรียลไทม์ระหว่างภาพอ้างอิงและเฟรมที่จับภาพจากเว็บแคม สคริปต์เริ่มต้นด้วยการนำเข้าห้องสมุดที่จำเป็น ได้แก่ numpy และ cv2 (OpenCV) จากนั้นจะทำการเปิดการจับภาพจากเว็บแคมโดยใช้ cv2.VideoCapture(0)

? สคริปต์ตั้งค่า SIFT (Scale-Invariant Feature Transform) detector และ brute-force matcher โดยใช้ cv2.SIFT_create() และ cv2.BFMatcher() ตามลำดับ เครื่องมือเหล่านี้ใช้ในการตรวจจับและจับคู่จุดเด่นระหว่างภาพ ภาพอ้างอิง "conan/conan1.jpg" ถูกอ่านในโหมดสีเทาและปรับขนาดให้เท่ากับขนาดเดิม

? จุดเด่นและตัวบรรยายของภาพอ้างอิงถูกคำนวณโดยใช้ SIFT detector ด้วย detector.detectAndCompute(ref, None) สคริปต์จะเข้าสู่ลูปที่ไม่มีที่สิ้นสุดซึ่งจะจับภาพจากเว็บแคมอย่างต่อเนื่อง ตรวจจับจุดเด่นและตัวบรรยายในแต่ละเฟรม และจับคู่กับภาพอ้างอิงโดยใช้ brute-force matcher กับ k-nearest neighbors (matcher.knnMatch(des1, des2, k=2))

? สคริปต์จะกรองการจับคู่ที่ดีโดยใช้การทดสอบอัตราส่วนระยะทาง ซึ่งการจับคู่จะถือว่าดีถ้าระยะทางของการจับคู่ที่ใกล้ที่สุดน้อยกว่า 60% ของระยะทางของการจับคู่ที่ใกล้ที่สุดอันดับสอง การจับคู่ที่ดีเหล่านี้จะถูกวาดบนเฟรมโดยใช้ cv2.drawMatches() และแสดงในหน้าต่างชื่อ "result"

? ลูปจะดำเนินต่อไปจนกว่าผู้ใช้จะกดปุ่ม 'q' จากนั้นการจับภาพจะถูกปล่อยและหน้าต่าง OpenCV ทั้งหมดจะถูกปิดโดยใช้ cap.release() และ cv2.destroyAllWindows() สคริปต์นี้แสดงให้เห็นถึงการจับคู่คุณลักษณะและการแสดงผลแบบเรียลไทม์โดยใช้ OpenCV
"""

cap = cv2.VideoCapture(0)

detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()  # ? brute-force matcher

# detector = cv2.ORB_create()
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

ref = cv2.imread("conan/conan1.jpg", cv2.COLOR_BGR2GRAY)
h, w, _ = ref.shape
ref = cv2.resize(ref, (int(w * 1.0), int(h * 1.0)))

kp1, des1 = detector.detectAndCompute(ref, None)

while True:
    _, target = cap.read()

    kp2, des2 = detector.detectAndCompute(target, None)

    matches = matcher.knnMatch(des1, des2, k=2)  # Relative Matching

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    result = cv2.drawMatches(ref, kp1, target, kp2, good, None, flags=2)

    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
