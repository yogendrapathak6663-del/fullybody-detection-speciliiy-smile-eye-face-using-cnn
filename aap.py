import cv2

face_cascade=cv2.CascadeClassifier(r"edgeDection\facedetection.py\haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    #now convert the face into the blackwhite for best view
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray,1.1,5)

    """multiscale detect the object"""

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        """
        x- how far from left
        y-how far from top
        w-width of the image
        z-height of faces
        """

    cv2.imshow("Webcame face detection",frame)

    if cv2.waitKey(1) & 0xFF==ord('e'):
        break
cap.release()
cv2.destroyAllWindows()       