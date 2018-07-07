import numpy as np
import cv2
path1 = "haarcascade_frontalface_default.xml"
path2 = "haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier(path1)
eye_cascade = cv2.CascadeClassifier(path2)
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Face Detect", frame)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
    eye = eye_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=20, minSize=(10,10))
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Face Detect", frame)
    for (x, y, w, h) in eye:
        xc = (x + x+w)/2
        yc = (y + y+h)/2
        radius = w/2
        cv2.circle(frame, (int(xc),int(yc)), int(radius), (0,255,0), 2);
    cv2.imshow("Face Detect", frame)
    ch = cv2.waitKey(1)
    if (ch & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
