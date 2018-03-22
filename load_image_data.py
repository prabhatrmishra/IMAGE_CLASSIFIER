import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os

label = os.listdir("dataset_image")
dataset=[]
for image_label in label:

    images = os.listdir("dataset_image/"+image_label)                  # Load all the data

    for image in images:
        img = cv2.imread("dataset_image/"+image_label+"/"+image)
        img = cv2.resize(img, (64, 64))
        dataset.append((img,image_label))

X=[]
Y=[]

for  input_image,image_label in dataset:

    X.append(input_image)                          #integer encode the labels
    Y.append(label.index(image_label))

X=np.array(X)
Y=np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=7)

#one got encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

data_set_train=(X_train,y_train)
data_set_test=(X_test,y_test)
