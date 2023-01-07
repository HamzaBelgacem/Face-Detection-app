import numpy as np 
import cv2
cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("C:/Users/Hamza/Desktop/ItGate/project face detectation/haarcascade_frontalface_default.xml")
id=input("donner user Id :")
num=0
while (True) :
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (70,255,255),5)
        num+=1
        cv2.imwrite("dataset/user."+str(id)+"."+str(num)+".jpg",gray[y:y+h,x:x+w])
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif num>30:
        break
    
cap.release()
cv2.destroyAllWindows()