import face_recognition
import cv2
import numpy

img1 = face_recognition.load_image_file("./known_faces/elon.jpg")
encodings1=face_recognition.face_encodings(img1)
if encodings1:
    encoding1=encodings1[0]
else:
    print("no face found")
    exit()

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("error")
    exit()

while True:
    success, frame=cap.read()
    if not success:
        break
    
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces=face_cascade.detectMultiScale(rgb,scaleFactor=1.1, minNeighbors=5)

    for(x,y,h,w) in faces:
        cv2.rectangle(frame, (x,y),(x+h,y+w),(0,255,0),2)

    cv2.imshow('frame',frame)

    face_loc=face_recognition.face_locations(rgb)
    face_encodings=face_recognition.face_encodings(rgb,face_loc)
    
    
    for encoding2 in face_encodings:
        results=face_recognition.compare_faces([encoding1],encoding2)

        if results[0]:
            print("same")
        else:
            print("different")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
