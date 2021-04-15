import numpy as np
import cv2
import time

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputfiles = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0
for x in range(60):
    ret,bg = cap.read()
    bg = np.flip(bg,axis = 1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    img = np.flip(img,axis = 1)

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lowerred = np.array(0,120,50)
upperred = np.array(10,255,255)
mask1 = cv2.inrange(hsv,lowerred,upperred)

lowerred1 = np.array(170,120,70)
upperred1 = np.array(180,255,255)
mask2 = cv2.inrange(hsv,lowerred1,upperred1)

mask = mask1+mask2
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
mask3 = cv2.bitwise_not(mask)

res1 = cv2.bitwise_and(img,img,mask = mask3)
res2 = cv2.bitwise_and(bg,bg,mask = mask3)

finaloutput = cv2.addWeighted(res1,1,res2,1,0)
outputfiles.write(finaloutput)
cv2.imshow('magic',finaloutput)
cv2.waitKey(1)
cap.release()
out.release()
cv2.destroyAllWindows()
