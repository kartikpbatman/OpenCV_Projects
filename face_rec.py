import cv2
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error')
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x,y,h,w) in faces:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),2)

    cv2.imshow('pr',frame)

    if cv2.waitKey(1) == ord('q'):
        exit()