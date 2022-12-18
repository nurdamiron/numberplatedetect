import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import imutils

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

top = tk.Tk()
top.geometry('1366x720')
top.title('Number Plate Recognition')
load = Image.open('background.jpg')
render = ImageTk.PhotoImage(load)
img = Label(top,image=render)
img.place(x=0,y=0)

top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 35, 'bold'))
sign_image = Label(top)
plate_image_ocv = Label(top)
plate_image_cnn = Label(top)

def classify_opencv(file_path):
    img = cv2.imread(file_path)

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    imgblur = cv2.GaussianBlur(img, (5, 5), 0)
    imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
    imggray = cv2.bilateralFilter(imggray, 13, 15, 15)
    canny_edge = cv2.Canny(imggray, 150, 170)
    contours = cv2.findContours(canny_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    cont_lic_plate = None
    lic_plate = None
    x, y, w, h = None, None, None, None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)  # measures the length means perimeter
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # detecting the shapes
        if len(approx) == 4:  # license plate is rectangular so vertices should be 4
            cont_lic_plate = approx
            x, y, w, h = cv2.boundingRect(contour)  # it rectifies the points like height,width,x,y coordinates
            blk = np.zeros(img.shape,np.uint8)
            cv2.rectangle(blk,(x,y),(x+w,y+h),(0,255,0),cv2.FILLED)
            img = cv2.addWeighted(img,1.0,blk,0.7,1)
            cv2.putText(img, "NUMBER PLATE", (x, y - 5), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            break
    if cont_lic_plate is None:
        detected = 0
        print("No contour detected")

        uploaded = Image.open("car.jpg")
        uploaded = uploaded.resize((350, 250), Image.ANTIALIAS)
        ###############################

        im = ImageTk.PhotoImage(uploaded)
        plate_image_ocv.configure(image=im)
        plate_image_ocv.image = im
        plate_image_ocv.pack()
        plate_image_ocv.place(x=520, y=110)


    else:
        detected = 1
    mask = np.zeros(imggray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [cont_lic_plate], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(file_path + 'result.png', img)
    # cv2.waitKey(0)

    uploaded = Image.open(file_path + "result.png")
    uploaded = uploaded.resize((350, 250), Image.ANTIALIAS)

    im = ImageTk.PhotoImage(uploaded)
    plate_image_ocv.configure(image=im)
    plate_image_ocv.image = im
    plate_image_ocv.pack()
    plate_image_ocv.place(x=520, y=110)


def classify_cnn(filepath):
    myTransformer = keras.models.load_model("model.h5")

    dataFrame = {
        'image': []
    }

    filepath1 = filepath
    size = 256
    img = plt.imread(filepath)
    if (filepath1.find('.jpg') != -1):
        cv2.imwrite(filepath+"test_img.png", img)
        img = plt.imread(filepath+'test_img.png')
    else:
        img = plt.imread(filepath)

    img = np.array(img[:, :, 0:3])
    dataFrame['image'].append(cv2.resize(img, (size, size)))
    plt.imshow(dataFrame['image'][0])
    plt.title('Input Image')
    Prediction, actuals = predict16(dataFrame, myTransformer)
    Plotter(actuals[0], Prediction[0],filepath)




def predict16(valMap, model, shape=256):
    img = valMap['image']
    imgProc = np.array(img)
    predictions = model.predict(imgProc)
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))

    return predictions, imgProc


def Plotter(img, predMask,file_path):
    plt.title('Predicted Box position')
    cv2.imwrite(file_path+'carmask1.png', predMask*255)
    uploaded2 = Image.open(file_path+"carmask1.png")
    uploaded2 = uploaded2.resize((350, 250), Image.ANTIALIAS)
    im2 = ImageTk.PhotoImage(uploaded2)
    plate_image_cnn.configure(image=im2)
    plate_image_cnn.image = im2
    plate_image_cnn.pack()
    plate_image_cnn.place(x=940, y=110)



def show_classify_button_opencv(file_path):
    classify_b = Button(top, text="Detect OpenCV", command=lambda: classify_opencv(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 15, 'bold'))
    classify_b.place(x=620, y=550)

def show_classify_button_cnn(file_path):
    classify_b = Button(top, text="Detect CNN", command=lambda: classify_cnn(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 15, 'bold'))
    classify_b.place(x=1050, y=550)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        file_path1 = file_path
        file_path2 = file_path

        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))

        uploaded = uploaded.resize((350, 250), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        # label.configure(text='')
        show_classify_button_opencv(file_path1)
        show_classify_button_cnn(file_path2)

    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 15, 'bold'))
upload.pack()
upload.place(x=190, y=550)
sign_image.pack()
sign_image.place(x=100, y=110)

heading = Label(top)
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
