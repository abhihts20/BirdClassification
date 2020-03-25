import sys
from inspect import k
import numpy as np
import cv2
import pandas as pd
from PIL import ImageTk,Image
import tkinter as tk

from mainfile import classifier
path="C:\\Users\\Abhinav\\Downloads\\test.jpg"
im = cv2.imread(path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.resize(im, (28, 28))

data = []
for i1 in range(0, 28):
    for j1 in range(0, 28):
        data.append(im[i1][j1])

print(classifier.predict([data]))
window = tk.Tk()
window.title("Join")
window.geometry("300x300")
window.configure(background='grey')
name=classifier.predict([data])
print(classifier.predict([data]))
img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image = img)
panel1=tk.Label(window,text=name)
panel.pack(side = "bottom", fill = "both", expand = "yes")
panel1.pack()
window.mainloop()


# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 6)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # bottom left and top right coordinate
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = img[y:y + h, x:x + w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#     cv2.imshow('img', img)
#     cv2.putText(img,model.predict([data]),(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()