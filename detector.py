import cv2
import numpy as np
import pickle
import sqlite3

#Nhập file train sẵn để phát hiện khuôn mặt
faceDetect=cv2.CascadeClassifier('./recognizer/haarcascade_frontalface_default.xml')

#Khởi động Camera
cam=cv2.VideoCapture(0)

#Khởi tạo thư viện training dữ liệu
rec=cv2.face.LBPHFaceRecognizer_create()

#Đọc file đã training
rec.read("./recognizer/trainningData.yml")

id=0
#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (203,23,252)

#Hàm xử lý khi detect thành công
def getProfile(id):
    print(id)
    if id==0: return 'Khong'
    else: return 'Co'
    

while(True):
    #img là hình ảnh màu chụp từ camera.
    ret,img=cam.read()

    #Biến ảnh màu thành trắng đen
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Phát hiện khuôn mặt trong hình bằng file train sẵn phía trên
    faces=faceDetect.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    #Duyệt tất cả khuôn mặt trong hình
    for(x,y,w,h) in faces:
        #Vẽ khung hình vuông trong ảnh
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #Dectect ra khuôn mặt bằng hàm predict
        id,conf=rec.predict(gray[(y+y+h)//2:y+h,x:x+w])

        #Gọi hàm xử lý khi dectect thành công
        profile=getProfile(id)

        #Vẽ tên lên hình
        if(profile!=None):
            cv2.putText(img, "Name: " + str(profile), (x,y+h+30), fontface, fontscale, fontcolor ,2)
        
        #Hiện hỉnh ảnh xem trước
        cv2.imshow('Face',img)

    #Thoát nếu nhấn phím 'q'
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

#Giải phóng thư viện camera
cam.release()
cv2.destroyAllWindows()
