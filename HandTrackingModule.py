import cv2 as cv
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self,mode=False,maxHands=2,modelComplexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.modelComplexity=modelComplexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.color=(0,255,0)
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.modelComplexity,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        self.DrawSpecs=self.mpDraw.DrawingSpec(color=self.color)
        self.tipIds=[4,8,12,16,20]

    def setColor(self,color=(255,255,255)):
        self.color=color
        self.DrawSpecs=self.mpDraw.DrawingSpec(color=self.color)

    def findHands(self,img,draw=True):
        self.DrawSpecs=self.mpDraw.DrawingSpec(color=self.color)
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS,self.DrawSpecs,self.DrawSpecs)
        return img
    
    def findPosition(self,img,handNo=0,draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                # print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id,cx,cy])
                # print(id,cx,cy)
                # if id==4:
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,255),-1)
            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,ymin,xmax,ymax
            if draw:
                cv.rectangle(img,(xmin-20,ymin-20),(xmax+20,ymax+20),(0,255,0),2)
        return self.lmList,bbox
    
    def fingersUp(self):
        fingers=[]
        u=1
        v=1
        if self.lmList[5][1]<self.lmList[17][1]:
            u=-1
        if self.lmList[0][2]<self.lmList[9][2]:
            v=-1
        for i in range(4,21,4):
            if(i==4):
                if(u*self.lmList[i][1]>u*self.lmList[i-2][1]):
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if(v*self.lmList[i][2]<v*self.lmList[i-2][2]):
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def findDistance(self,p1,p2,img,draw=True,r=10,t=3):
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        if draw:
            cv. line (img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 150), cv. FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 150), cv.FILLED)
            cv.circle(img,(cx,cy),r,(0,0,255),-1)
        
        length=math.hypot(x2-x1,y2-y1)
        return length,img,[x1,y1,x2,y2,cx,cy]

def main():
    cTime=0
    pTime=0
    capture=cv.VideoCapture(0)
    detector=HandDetector()
    while(True):
        sucess,img=capture.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        
        cv.imshow("Image",img)
        if cv.waitKey(20) & 0xff==ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


