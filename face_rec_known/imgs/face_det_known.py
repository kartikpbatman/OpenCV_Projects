import cv2
import face_recognition
import os

img1=face_recognition.load_image_file("unk.jpg")
encodings1 = face_recognition.face_encodings(img1)
if encodings1:
    encoding1 = encodings1[0]
else:
    print("No face found in img1.")
    exit()


img2=face_recognition.load_image_file("unk5.jpg")
encodings2 = face_recognition.face_encodings(img1)
if encodings2:
    encoding2 = encodings2[0]
else:
    print("No face found in img2.")
    exit()


results=face_recognition.compare_faces([encoding1],encoding2)

if results[0]:
    print("same")

else:
    print("different")
