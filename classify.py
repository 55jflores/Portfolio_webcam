import torch
from torchvision import transforms 

import cv2

# Loading in model
device = torch.device("cpu")
model = torch.jit.load('glasses_or_not_torchScript_model.pt').to(device)

# Transformation to be done on image
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

# Class names
class_names = ['Glasses','No_Glasses']

face_cascade = cv2.CascadeClassifier('haar_face.xml')
def detect_face(image):
    face_image = image.copy()

    # Detecting faces
    faces = face_cascade.detectMultiScale(face_image,scaleFactor=1.1,minNeighbors=10)

    # Resizing detected face to 170x170 
    if len(faces) == 1:
        for(x,y,w,h) in faces:
            face_box = face_image[y:y+h,x:x+w]
        return face_box
    return 'None'

def predict(image,face_image):
    # Predicting
    pred = model(face_image)

    c_prob = torch.exp(torch.tensor(pred[0][0].item()))
    d_prob = torch.exp(torch.tensor(pred[0][1].item()))

    prob = c_prob if c_prob > d_prob else d_prob 
    prob = round(100*prob.item(),2)
    cv2.putText(image,f'{class_names[pred.argmax().item()]}: {prob}%',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    #print('Model choose:',class_names[pred.argmax().item()],'\n')

    return image

cap = cv2.VideoCapture(0)
while True:
    # Read in frame
    success,image = cap.read()
    # using haarcascade to extract face
    face_image = detect_face(image)
    if str(face_image) != 'None':
        face_image = cv2.resize(face_image,(170,170))
        face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2RGB)
        # Converting to tensor and normalizing
        face_image = test_transform(face_image)
        # Reshaping for model
        face_image = face_image.view(1,3,170,170)
        # Doing prediction
        image = predict(image,face_image)

    cv2.imshow('Webcam',image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()