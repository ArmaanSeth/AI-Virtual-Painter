import numpy as np
import time
import cvzone
import cv2 as cv
import os
import HandTrackingModule as htm


ctime=0
ptime=0
smoothening=3

path=os.path.dirname(__file__)+'/img'
myList=os.listdir(path)

overlayList=[]
for img in myList:
    impath=f'{path}/{img}'
    overlayList.append(cv.imread(impath,cv.IMREAD_UNCHANGED))
header=overlayList[1]

cap=cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector=htm.HandDetector(detectionCon=0.75)
color=(255,0,0)
xp=0
yp=0
brushThickness=15
eraserThickness=50
imgCanvas=np.zeros((720,1280,3),np.uint8)

while(True):
    success,img=cap.read()
    img=cv.flip(img,1)

    image=detector.findHands(img)
    lmList,bbox=detector.findPosition(img,False)

    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        # Which fingers are up
        fingers=detector.fingersUp()

        # Selection Mode when two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection Mode`")
            xp=0
            yp=0
        
            if y1<160:
                if 320<x1<480:
                    color=(255,0,0)
                    header=overlayList[1]    
                elif 560<x1<715:
                    color=(0,255,0)
                    header=overlayList[2]
                elif 800<x1<950:
                    color=(0,0,255)
                    header=overlayList[3]
                elif 1045<x1<1180:
                    color=(0,0,0)
                    header=overlayList[4]
            detector.setColor(color)
            cv.rectangle(img,(x1,y1),(x1+brushThickness,y1+brushThickness),color,-1)
        
        # Drawing Mode when index finger is up
        if fingers[1] and fingers[2]==False:
            print("Drawing Mode")
            cv.circle(img,(x1,y1),10,color,-1)
            
            if xp==0 and yp==0:
                xp=x1
                yp=y1
            
            x1=xp+int((x1-xp)/smoothening)
            y1=yp+int((y1-yp)/smoothening)
            
            if color==(0,0,0):
                cv.line(img,(xp,yp),(x1,y1),color,eraserThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),color,eraserThickness)    
                cv.circle(img,(x1,y1),int(eraserThickness/2),color,-1)
            else:
                cv.line(img,(xp,yp),(x1,y1),color,brushThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),color,brushThickness)
                cv.circle(img,(x1,y1),int(brushThickness/2),color,-1)
            xp=x1
            yp=y1
            
    else:
        xp=0
        yp=0

    imgGrey=cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _,imgInv=cv.threshold(imgGrey,0,255,cv.THRESH_BINARY_INV)
    imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img=cv.bitwise_and(img,imgInv)
    img=cv.bitwise_or(img,imgCanvas)

    img=cvzone.overlayPNG(img,header,[0,0])
    # ctime=time.time()
    # fps=1/(ctime-ptime)
    # ptime=ctime
    # cv.putText(img,f'FPS:{int(fps)}',(500,50),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    cv.imshow("AI Virtual Painter",img)
    # cv.imshow("Canvas",imgCanvas)
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
