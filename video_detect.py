# This code is to simply understand how our webcam is able to capture the video and show it.
# Also it tells how the video is changed to grayscale to perform operations.
import cv2, time
a=1
video=cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    a=a+1
    check, frame=video.read()
    print(check)
    print(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture",gray)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
print(a)

video.release()
cv2.destroyAllWindows()
