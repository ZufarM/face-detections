import cv2
import os
import numpy as np
from PIL import Image

wajah_dir = 'face_dataset'
latih_dir = 'face_trained'

def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    i = 0
    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L') #convert to grey
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].replace("face.", "").split(".")[0])
        print(imagePath)
        faces = face_detector.detectMultiScale(imgNum)
        i += 1
        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y + h, x:x + w])
            faceIDs.append(faceID)
            # print(faceIDs)
    return faceSamples, faceIDs

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

print("Mesin sedang melakukan training wajah")
faces, IDs = getImageLabel(wajah_dir)

face_recognizer.train(faces, np.array(IDs))
face_recognizer.write(latih_dir+'/training.xml')
print('Sebanyak '+format(len(np.unique(IDs)))+' data wajah telah ditrainingkan ke mesin.')