from ultralytics import YOLO
import cv2 as cv
import cvzone as cz
import math
from sort import *

cap = cv.VideoCapture("video/Example01.mp4")
model = YOLO("YoLo_Weight/yolov8n.pt")
className = ["person",'bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

mask = cv.imread('mask/mask.png')
mask = cv.resize(mask,(852,480))

#Tracking
tracker = Sort(max_age=20,min_hits=3, iou_threshold=0.3)

#line
limits = [150,297,600,297]
total_Cars = []
while True:
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img,mask)
    result = model (imgRegion,stream=True)
    
    detections = np.empty((0,5))
    
    for r in result:
        boxes = r.boxes
        for box in boxes:
            #Bouding
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)  
            w,h = x2-x1,y2-y1
            conf = math.ceil((box.conf[0])*100)/100
            #class
            cls = int(box.cls[0])
            currentClass= className[cls]
            if (currentClass == 'car' or currentClass == 'motorbike' or currentClass == 'truck' or currentClass == 'bus') and conf>0.3:
                #cz.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),scale=0.6, thickness=1,offset=3)
                #cz.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    resultTracker = tracker.update(detections)
    cv.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)
        w,h = x2-x1,y2-y1
        cz.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,255))
        cz.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale=2, thickness=3,offset=10)
        cx,cy = x1+w//2,y1+h//2
        cv.circle(img,(cx,cy),5,(255,0,255),cv.FILLED)
        
        if limits[0] < cx <limits[2] and limits[1]-15 <cy <limits[1]+15 :
            if total_Cars.count(id) == 0:
                total_Cars.append(id)
                cv.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
    cz.putTextRect(img,f' Count: {len(total_Cars)}',(50,50))
    cv.imshow("Image",img)
    if cv.waitKey(1) == 32:
        break
    
cap.release()
cv.destroyAllWindows()
