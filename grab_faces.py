import cv2
import numpy as np

# Use haarcascade to grab faces from webcam video recording
# Two videos: One without my glasses on and one with my glasses on
# Pytorch: Train it to detect if i have my glasses on or not

i = 587
face_cascade = cv2.CascadeClassifier('haar_face.xml')
def detect_face(image):
    global i 
    face_image = image.copy()

    # Detecting faces
    faces = face_cascade.detectMultiScale(face_image,scaleFactor=1.1,minNeighbors=10)

    # Drawing rectangle around face
    if len(faces) == 1:
        i = i + 1
        for(x,y,w,h) in faces:
            face_box_resized = cv2.resize(face_image[y:y+h,x:x+w],(170,170))
            cv2.imwrite("no_glasses%d.jpg" % i,face_box_resized)
            #cv2.rectangle(face_image,(x,y),(x+w,y+h),(255,0,0),2)
        return face_image
    return 'None'

cap = cv2.VideoCapture(0)
while True:
    success,image = cap.read()
    detected_image = detect_face(image)
    if str(detected_image) != 'None':
        image = detected_image
    cv2.imshow('Webcam',image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()