import imp
from operator import imod
import sys
from turtle import st
import cv2 #import modul
import os
from pathlib import Path

2
 # WajahDir = 'DataWajah'
image_dir = "DataWajah"

cam = cv2.VideoCapture(0)#mendeteksi kamera laptop
cam.set(3, 648)
cam.set(4, 488)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

faceID = [10]
nama = input("Masukan Nama ")
faceID = input("Masukan ID [Tekan Enter] : " )
path = os.path.join(image_dir, faceID+"_"+nama)

if(Path('DataWajah'+'/'+faceID+nama).is_dir()==False):
    os.mkdir(path)
WajahDir = 'DataWajah'+'/'+faceID+"_"+nama
        

print ("Tatap Wajah Anda ke depan dalam Webcam tunggu proses sampai selesai..")

takeData = 1
while True:
    retV, frame = cam.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        grey,
        scaleFactor = 1.5,
        minNeighbors = 2)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        namaFile = 'wajah.'+str(faceID)+str(nama)+'.'+str(takeData)+'.jpg'
        cv2.imwrite(WajahDir+'/'+namaFile,grey[y:y+h, x:x+w])
        takeData+= 1
        roi_grey = grey[y:y+h, x:x+w]
        roi_warna = frame[y:y+h, x:x+w]
        eye = eyeDetector.detectMultiScale(roi_grey)

        for(xe, ye, we, he) in eye:
            cv2.rectangle(roi_warna, (xe, ye), (xe+we, ye+he), (0,0,255),3)

    cv2.imshow('Webcam', frame)
    cv2.imshow('Webcam-Hitam Putih', grey)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('x'):
        break
    elif takeData>500:
        break

print("Pengambilan Data Wajah Selesai")

cam.release()
cv2.destroyAllWindows()


    