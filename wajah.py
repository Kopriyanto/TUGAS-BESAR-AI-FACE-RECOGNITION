from cProfile import label
from cgitb import grey, handler
import os
import tkinter as tk
from tkinter import *
import numpy as np
from turtle import onclick
import cv2
from PIL import Image, ImageTk
from matplotlib.pyplot import get, text
from setuptools import Command
from pathlib import Path
import pickle as pickle

root = tk.Tk()
root.title("Deteksi Wajah")
root.geometry('640x550')

label = Label(root)
label.grid(row=1, column=0)
cam = cv2.VideoCapture(0)
image_dir = 'DataWajah'
a = 20

class cap:
    def __init__(self, faceID):
        self.faceID = faceID

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

latihDir = 'latihwajah'

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(latihDir+ '/' + 'latih.xml')
        

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
# for a in range(20):
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        frame = cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        font = cv2.FONT_HERSHEY_SIMPLEX
        eyes = eye_detector.detectMultiScale(roi_gray)
        color = (255,255,255)
        stroke = 2

        id_, conf = recognizer.predict(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if conf >=45:
            # print(id_)
            # print(labels[id_])
            name = labels[id_]
            confidenceTxt = "{0}%".format(round(100-conf))
        else:
            # print("Tidak Terdaftar")
            # print(labels[id_])
            name = ("Tidak Terdaftar")
            confidenceTxt = "{0}%".format(round(100-conf))

        cv2.putText(frame, str(name), (x-50,y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.putText(frame,str(confidenceTxt),(x+5,y+h-5),font,1,(255,255,0),1)  

    img = Image.fromarray(frame)
    Imgtk = ImageTk.PhotoImage(image=img)
    label.Imgtk = Imgtk
    label.configure(image=Imgtk)
    root.update()







    




