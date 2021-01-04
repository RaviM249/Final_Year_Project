import numpy as np
import cv2
import pickle
import pyttsx3
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')# print(voices) to get the object    
engine.setProperty('voice',voices[0].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={}

with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for x,y,w,h in faces:
        # print(x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        id_, conf=recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            # print(id_)
            # print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            speak(name)
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        else:
            font=cv2.FONT_HERSHEY_SIMPLEX
            name="unknown"
            speak("unknown")
            color=(0,0,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img_item="my-image.png"
        cv2.imwrite(img_item, roi_color)
        color=(0,255,0)
        stroke=3
        end_x=x+w
        end_y=y+h
        cv2.rectangle(frame,(x,y),(end_x,end_y),color,stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()