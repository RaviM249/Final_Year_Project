import cv2
import pyttsx3
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')# print(voices) to get the object    
engine.setProperty('voice',voices[0].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_counter_prev=0
while True:
    _, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    try:
        face_counter_curr=faces.shape[0]
    except:
        face_counter_curr=0    
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
    if(face_counter_curr!= face_counter_prev):
        print(face_counter_curr, "faces detected")
        # speak(str(face_counter_curr)+ "faces detected")
        face_counter_prev=face_counter_curr
    
cap.release()