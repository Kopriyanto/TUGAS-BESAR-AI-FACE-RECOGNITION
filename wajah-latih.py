from asyncio import AbstractEventLoop
from cProfile import label
from PIL import Image
import numpy as np
from operator import ipow
import cv2
import pickle
import os
from matplotlib import image

from numpy import imag

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imag_dir = os.path.join(BASE_DIR, "DataWajah")
latihDir= 'latihwajah'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognition = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(imag_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)
            if not label in label_ids:
                label_ids[label]= current_id
                current_id += 1
            
            id_ = label_ids[label]
            print(label_ids)
            
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors=5)

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognition.train(x_train, np.array(y_labels))
recognition.save(latihDir+'/latih.xml')
