# Chapter 1
# Read Images - Videos - WebCam
import cv2

# Read WebCam OpenCV
cap = cv2.VideoCapture(0) # <- parameter adalah ID WebCam
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)

# Read Video OpenCV
# cap = cv2.VideoCapture('Resources/sample.mp4')

# While Video or WebCam
while True:
    success, img = cap.read()
    cv2.imshow('Video', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Read Image OpenCV
# img = cv2.imread('Resources/lena.png')
#
# cv2.imshow('Output', img)
# cv2.waitKey(0)