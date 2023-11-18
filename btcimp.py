#!/usr/bin/env python
# coding: utf-8

# In[55]:


pip install tensorflow==2.12.0


# In[56]:


pip install opencv-python


# In[57]:


pip install numpy==1.21.6


# In[59]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
import os




# In[60]:


from sklearn.metrics import accuracy_score


# In[61]:


import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np


# In[62]:


X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join(r'C:\Users\Navya Nagesh\Downloads\Telegram Desktop\archive (3)\Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
        
for i in labels:
    folderPath = os.path.join(r'C:\Users\Navya Nagesh\Downloads\Telegram Desktop\archive (3)\Testing',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# In[63]:


X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_train.shape


# In[64]:


X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)


# In[65]:


y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)


# In[66]:


model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))


# In[67]:


model.summary()


# In[68]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[27]:


history = model.fit(X_train,y_train,epochs=20,validation_split=0.1)


# In[69]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[70]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(11,3))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()


# In[71]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
fig = plt.figure(figsize=(11,3))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc='upper left')
plt.show()


# In[81]:


img = cv2.imread(r'C:\Users\Navya Nagesh\AppData\Local\Temp\6502eb58-2a1f-4d71-a724-3d602e883de7_archive (3).zip.de7\Testing\glioma_tumor\image(1).jpg')
img = cv2.resize(img,(150,150))
img_array = np.array(img)
img_array.shape


# In[82]:


img_array = img_array.reshape(1,150,150,3)
img_array.shape


# In[83]:


from tensorflow.keras.preprocessing import image
img = image.load_img(r'C:\Users\Navya Nagesh\AppData\Local\Temp\6502eb58-2a1f-4d71-a724-3d602e883de7_archive (3).zip.de7\Testing\glioma_tumor\image(1).jpg')
plt.imshow(img,interpolation='nearest')
plt.show()


# In[84]:


a=model.predict(img_array)
indices = a.argmax()
indices

