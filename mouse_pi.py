import cv2 as cv
import numpy as np
from pynput.mouse import Button, Controller
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import wx

pinchFlag=0
mouse=Controller()  #Mouse object

app=wx.App(False)   #initializing app to get display coordinates only
(sx,sy)=wx.GetDisplaySize() #Display Monitor Coordinates
(camx,camy)=(320,240)   #Cam resolution

mLocOld=np.array([0,0])
mouseLoc=np.array([0,0])
DampingFactor=4     #Make mouse move less violent


lowerBound=np.array([40,80,40])
upperBound=np.array([102,255,255])  #HSV Color bounds
#yellow_lwr = np.uint8([[[0,255,0 ]]])
#yellow_upr = np.uint8([[[0,255,0 ]]])
#hsv_yellow_lower = cv.cvtColor(yellow_lwr,cv.COLOR_BGR2HSV)
#hsv_yellow_upper = cv.cvtColor(yellow_upr, cv.COLOR_BGR2HSV)
cam=cv.VideoCapture(0)
# cam.set(3,camx) #3 is width flag
# cam.set(4,camy) #4 is height flag
kernelOpen=np.ones((5,5))   #Morphology opening kernel
kernelClose=np.ones((8,8))    #Morphology closing kernel


while True:
    #ret, img=cam.read()
    camera = PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(320, 240))
    # allow the camera to warmup
    time.sleep(0.1)
    #if ret:
    #	img=cv.resize(img,(320,240))   #Resizing image for faster processing
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
    	img = frame.array
    	imgHSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)    #Convert RGB to HSV

        mask=cv.inRange(imgHSV,lowerBound,upperBound)  #mask with no Morphology or noise screening applied

      	maskOpen=cv.morphologyEx(mask,cv.MORPH_OPEN,kernelClose)   #Applying Morphology, removing outside noise being scanned from video
        	#maskClose=cv.morphologyEx(mask,cv.MORPH_CLOSE,kernelClose) #removing inner noise
       	maskFinal=maskOpen
       	_, conts, hier =cv.findContours(maskFinal.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)  #Obtaining countour box for the found object
        cv.drawContours(img,conts,-1,(0,0,255),3)      #drawing contour box on original image

        if(len(conts)==2):  #case for two rectangles
            if pinchFlag==1:
                pinchFlag=0
                mouse.release(Button.left)  #reducing mutiple clicks
            x1,y1,w1,h1=cv.boundingRect(conts[0])
            x2,y2,w2,h2=cv.boundingRect(conts[1])
            cv.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)  # Drawing contour rectangle for 1st box
            cv.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)  # Drawing contour rectangle for 1st box
            cx1, cy1 = int((x1 + w1 / 2)), int((y1 + h1 / 2))
            cx2, cy2 = int((x2 + w2 / 2)), int((y2 + h2 / 2))       #Finding middle point of both contours
            cv.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)   #Drawing a line between two contours
            cx,cy=int((cx1+cx2)/2),int((cy1+cy2)/2)
            cv.circle(img,(cx,cy),2,(0,0,255),2)   #plotting a center point between two lines
            mouseLoc=mLocOld+((cx,cy)-mLocOld)/DampingFactor
            mouse.position=(sx-(mouseLoc[0]*sx/camx), mouseLoc[1]*sy/camy)
            mLocOld=mouseLoc

        elif(len(conts)==1):   #case for a single rectangle
            if pinchFlag==0:
                pinchFlag=1
                mouse.press(Button.left)    #reducing mutiple clicks
            x,y,w,h=cv.boundingRect(conts[0])
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cx=int(x+w/2)
            cy=int(y+h/2)
            cv.circle(img,(cx,cy),int((w+h)/4),(0,0,255),2)    #plotting a big circle with centre as rectangle's centre
            mouseLoc = mLocOld + ((cx, cy) - mLocOld) / DampingFactor
            mouse.position = (sx - (mouseLoc[0] * sx / camx), mouseLoc[1] * sy / camy)
            mLocOld = mouseLoc
        rawCapture.truncate(0)
        cv.imshow("cam",img)   #Cam OP
        k = cv.waitKey(5)
        if (k==27):
            break;
