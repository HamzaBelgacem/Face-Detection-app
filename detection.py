import numpy as np 
import cv2
cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("C:/Users/Hamza/Desktop/ItGate/project face detectation/haarcascade_frontalface_default.xml")

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recongnizer/trainingData.yml')
id=0

while (True) :
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (70,255,255),5)
    
        id, conf=rec.predict(gray[y:y+h, x:x+w])
        
        if conf<50:

            if (id==1):
                id='hamza'
        else:
            id='inconnu'
        cv2.putText(frame, str(id),(x,y+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(frame, str(conf),(x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))


    cv2.imshow('reconnaissance faciale', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()