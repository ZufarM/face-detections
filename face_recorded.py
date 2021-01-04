# Face Detected
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640) # lebar cam
cap.set(4, 480) # tinggi cam
3
face_ID = input("Masukan NIM Mahasiswa yang akan direkam  [Kemudian tekan Enter]: ")
print("Tatap wajah anda ke Webcam & Tunggu hingga proses pengambilan data wajah selesai...")

wajah_dir = 'face_dataset'
ambil_data = 1

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        nama_file = 'face.'+str(face_ID)+'.'+str(ambil_data)+'.jpg'
        cv2.imwrite(wajah_dir+'/'+nama_file, frame)
        ambil_data += 1

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif ambil_data > 30:
        break
print("Pengambilan data selesai")
cap.release()
cv2.destroyAllWindows()