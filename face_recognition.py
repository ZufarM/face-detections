import cv2

wajah_dir = 'face_dataset'
latih_dir = 'face_trained'

cam = cv2.VideoCapture(0)
cam.set(3, 640) # lebar cam
cam.set(4, 480) # tinggi cam

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read(latih_dir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Diketahui', 'Dwi', 'Zufar']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while 1:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(round(minWidth), round(minHeight)))

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
        if confidence <= 50:
            nameID = names[id]
            confidenceTxt = " {0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = " {0}%".format(round(100-confidence))
        cv2.putText(frame, str(nameID), (x+5, y+5), font, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(confidenceTxt), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Face Recognition', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

print("EXIT")
cam.release()
cv2.destroyAllWindows()