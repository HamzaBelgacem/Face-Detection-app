import os 
import cv2
import numpy as np 
import PIL 
from PIL import Image
recognizer=cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'
def getImagesWithID(path):
    imagePaths=[os.path.join(path ,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:

        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg, 'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(ID)
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return IDs, faces 
IDs,faces=getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('C:/Users/Hamza/Desktop/ItGate/project face detectation/recongnizer/trainingData.yml')
cv2.destroyAllWindows()    

# cwd = os.getcwd()
# dataset = os.path.join(cwd, "C:/Users/Hamza/Desktop/ItGate/project face detectation/dataset")
# files = os.listdir(dataset)
# for f in files:
#     print(files)
#     print(os.path.join(dataset, f))
