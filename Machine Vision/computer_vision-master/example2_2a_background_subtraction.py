import cv2

#Download 'ExampleBGSubtraction.avi' from https://drive.google.com/file/d/1OD_A0wqN2Om2SusCztybu-_hMSUQuRt7/view?usp=sharing

cap = cv2.VideoCapture('ExampleBGSubtraction.avi')

haveFrame,bg = cap.read() # ? First frame is background

while(cap.isOpened()):
    haveFrame,im = cap.read()

    if (not haveFrame) or (cv2.waitKey(10) & 0xFF == ord('q')):
        break

    diffc = cv2.absdiff(im,bg) # ? background subtraction ( im<uint8> - bg<uint8> ) [cv2.adsdiff(im1, im2) is abs difference]
    diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY) # ? convert to grayscale
    bwmask = cv2.inRange(diffg,50,255) # ? threshold
    
    print(f"diffc : {diffc.shape}")
    print(f"diffg : {diffg.shape}")
    print(f"bwmask : {bwmask.shape}")

    cv2.imshow('diffc', diffc)
    cv2.moveWindow('diffc',10,10)
    cv2.imshow('diffg',diffg)
    cv2.moveWindow('diffg', 400, 10)
    cv2.imshow('bwmask', bwmask)
    cv2.moveWindow('bwmask', 800, 10)

cap.release()
cv2.destroyAllWindows()
