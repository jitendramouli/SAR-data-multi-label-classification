
#model for SAR data(sentinel 1) classification with tenserflow using optical data(sentinel 2)

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import random
from random import shuffle
from skimage.transform import rotate
import scipy.ndimage as ndimage
from PIL import Image
import glob
from scipy import misc
import spectral
import imageio
import h5py
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from keras.optimizers import SGD,Adadelta,Adam
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import itertools

K.set_image_data_format("channels_first")
from sklearn.preprocessing import OneHotEncoder 
from keras import losses
import cv2
import matplotlib.pyplot as plt


def loaddata():
    data_path=os.path.join(os.getcwd())
    data=imageio.imread(os.path.join(data_path,'sar.png'))
    labels=imageio.imread(os.path.join(data_path,'optical.png'))
    return data,labels

#converting an RGB image to a groundtruth classifier image i.e., x,y pixel location will have the class number
def label_creator(y):
    y_final = np.zeros([y.shape[0],y.shape[1]])
    #print(y_final.shape)
    cls = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j][0] == max(y[i][j]):
                cls = 1
            elif y[i][j][1] == max(y[i][j]):
                cls = 2
            else:
                cls = 3
            
            y_final[i][j] = cls
        if(i%100==0):print(i,end=' ')

    
    return y_final


def image_creatoror(img):
    pred_img = np.zeros((256,256,3))
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            if img[i][j]==1:
                pred_img [i][j]=(0,0,0)
            else:
                pred_img [i][j]=(10,104,10)
    return pred_img        

def image_creatortes(img):
    pred_img = np.zeros((256,256,3))
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            if img[i][j][0]>0.7:
                pred_img [i][j]=(0,0,0)
            elif img[i][j][1]>0.24:
                pred_img [i][j]=(10,104,10)
            else:
                pred_img[i][j]=(39,39,244)
    return pred_img        



def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


#Creating patches of 5x5x3 so that each patch contains 1 pixel at the centre, surrounded by zero padded pixels

def createPatches(X, y, windowSize, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def slicingindex(X):
    x = np.zeros((X.shape[0],X.shape[1],X.shape[2],3))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                    x[i][j][k][0]=0
                    x[i][j][k][1]=X[i][j][k][1]
                    x[i][j][k][2]=0    
        if(i%10000==0):print(int(i/10000),end=' => ')
    return x

def slicingindexy(X):
    x = np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x[i][j]=X[i][j][1]    
    return x


#model training and construction

channels=3
windowsize=9

x,k=loaddata()

y=y[:,:,:-1]

x=cv2.resize(x,(1163,2418))
plt.imshow(cv2.cvtColor(x,cv2.COLOR_BGR2RGB)) 
y=cv2.resize(y,(4000,8000))
plt.imshow(cv2.cvtColor(y,cv2.COLOR_BGR2RGB)) 


y=label_creator(y)

with open(os.getcwd()+'/Y_lable.npy','bw') as outfile: 
    np.save(outfile,y)
    
y=np.load(os.getcwd()+'/Y_lable.npy')
#y=y.astype(np.uint8)

Xpatches,ypatches=createPatches(x,y,windowSize=windowsize)

X_train,X_test,Y_train,Y_test=train_test_split(Xpatches,ypatches,test_size=0.50)

with open(os.getcwd()+'/X_train.npy','bw') as outfile:
    np.save(outfile,X_train)
with open(os.getcwd()+'/y_train.npy','bw') as outfile:
    np.save(outfile,Y_train)
with open(os.getcwd()+'/X_test.npy','bw') as outfile:
    np.save(outfile,X_train)
with open(os.getcwd()+'/y_test.npy','bw') as outfile:
    np.save(outfile,Y_train)
X_train=np.load(os.getcwd()+'/X_train.npy') 
Y_train=np.load(os.getcwd()+'/Y_train.npy') 
X_test=np.load(os.getcwd()+'/X_test.npy') 
Y_test=np.load(os.getcwd()+'/y_test.npy') 

#X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[3],X_test.shape[1],X_test.shape[2]))
#Xpatches=np.reshape(Xpatches,(Xpatches.shape[0],Xpatches.shape[3],Xpatches.shape[1],Xpatches.shape[2]))
#X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[3],X_train.shape[1],X_train.shape[2]))
y_train=to_categorical(Y_train)
input_shape=X_train[0].shape
#onehotencoder = OneHotEncoder(sparse=False) 
#Y_train.reshape(1,-1)
#Y_train = onehotencoder.fit_transform(Y_train).toarray() 


#input_shape = (256,256,3)
#WORKING BUT EACH EPOCH TAKES 1.5 HOURS FOR BATCH SIZE 1000

''' ------------------------------u net architecture--------------------------------- '''

model = Sequential()
model.add(Conv2D(1, (3, 3), activation='relu',padding='same',data_format='channels_first'))
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2),name='block1_pool',data_format='channels_first'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2),data_format='channels_first'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))#transposed convolution

model.add(Conv2D(1, (3, 3), activation='relu',padding='same'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=10000,epochs=5)

model.save(os.getcwd()+'/mymodel.h5')



X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[3],X_train.shape[1],X_train.shape[2]))


'''------------------------ mango net architecture---------------------------------- '''

model = Sequential()
model.add(Conv2D(32, kernel_size=(1,1),input_shape=input_shape,activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))


#model.add(Dropout(0.8))
model.add(Conv2D(32, kernel_size=(1,1),activation='relu',data_format='channels_first'))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.8))
model.add(Conv2D(32, kernel_size=(1,1),activation='relu',data_format='channels_first'))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(UpSampling2D(size=(1,1),data_format='channels_first'))
model.add(UpSampling2D(size=(3, 3),data_format='channels_first'))

#model.add(Dropout(0.8))
model.add(UpSampling2D(size=(3, 3),data_format='channels_first'))
model.add(Conv2D(27, (3, 3), activation='relu',padding='same',data_format='channels_first'))
model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(3, activation='softmax'))

#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

adam=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer = Adam(lr = 1e-4), metrics=['accuracy'])

model.fit(X_train,y_train,batch_size = 64,epochs=4)
#model.fit(X_train, Y_train, batch_size = 1000, nb_epoch = 5, verbose = 1, shuffle = True)

model.summary()

model.save(os.getcwd()+'/modks1.h5')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape,data_format='channels_first'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu',padding = 'same'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 1e-4), metrics=['accuracy'])


# ------------------testing with one image------------------------------

model = load_model(os.getcwd()+'/modks1.h5')

data = imageio.imread(os.getcwd()+'/image_23_3645496.png')
labels = imageio.imread(os.getcwd()+'/trail/trailop3.png')
X_test1 = data
y_test1 = label_creator(labels)


y_test1=image_creator(y_test1)
y_test1 = spectral.imshow(classes = y_test1.astype(int),figsize =(5,5))



X_test1,y_test1= createPatches(X_test1, y_test1, windowSize=5)

#X_test1  = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[3], X_test1.shape[1], X_test1.shape[2]))
Y_test1 = to_categorical(y_test1)

#model = load_model(os.getcwd()+'/mymodel.h5')
#model = load_model(os.getcwd()+'/model16.h5')

Y_pred = model.predict(X_test1)
#ypredox=Y_pred.astype(np.uint8)

#res1 = Image.fromarray(pred_img, mode = 'RGB')

#with open(os.path.join(os.getcwd(), 'pred.png'), 'wb') as f:
#    res1.save(f)

img = np.reshape(Y_pred,(256,256,3))

pred_img=image_creatortes(img)

predict_image = spectral.imshow(classes = pred_img.astype(int),figsize =(5,5))

#-------------------------------------------------------------------

"""
X_tra, X_tes, y_tra, y_tes = train_test_split( Y_pred,y_test1, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_tra, y_tra)
y_pred = regressor.predict(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

for i in range(img.shape[0]):
    for j in range(img.shape[0]):
        if img[i][j] == 1:
            pred_img[i][j] = (0,0,0)
        elif img[i][j]==2:
            pred_img[i][j] = (10, 104, 10)
        #else:
         #   pred_img[i][j]=(0,0,0)
        elif img[i][j] == 3:
            pred_img[i][j] == (39,39,244)
        else:
            pred_img[i][j] = (39,39,244)
"""

def reports (X_test,y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['land', 'coconut trees', 'water bodies']

    
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names,labels=range(len(y_test)))
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    
    return classification, confusion, Test_Loss, Test_accuracy


def avg_accuracy():
    total = 0
    loss = 0
    model = load_model(os.getcwd()+'/mymodel.h5')
    data_path = os.path.join(os.getcwd(),'test_sar')
    label_path = os.path.join(os.getcwd(),'test_optical')
    input_images = os.listdir(data_path)
    label_images = os.listdir(label_path)
    i = 0
    for image in input_images:
        #print(image.type)
        X_test = imageio.imread(os.getcwd()+'/test_sar/{}'.format(image))
        y_test = imageio.imread(os.getcwd()+'/test_optical/{}'.format(image))
        #print(y)
        y_test = label_creator(y_test)

        X_test,y_test= createPatches(X_test, y_test, windowSize=5)

        #X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[3], X_test.shape[1], X_test.shape[2]))
        y_test = to_categorical(y_test)

        os.chdir(os.getcwd())

        classification, confusion, Test_loss, Test_accuracy = reports(X_test,y_test)
        
        Y_pred = model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)

        img = np.zeros((256,256))
        
        j = 0
        for col in range(256):
            for row in range(256):
                img[col][row] = (y_pred[j])
                j+=1
                
        res = Image.fromarray(img, mode = 'RGB')
        with open(os.path.join(os.getcwd()+'/predicts', '{}'.format(image)), 'w') as f:
                     res.save(f)
            
        #predict_image = spectral.imshow(classes = img.astype(int),figsize =(5,5))
        spectral.save_rgb(os.getcwd()+'/pred/{}'.format(image),img)
        
        if i==0:
            acc = Test_accuracy
            loss = Test_loss
        acc = float((acc+Test_accuracy)/2)
        loss = float((loss+Test_loss)/2)
        print(i,'--------------')
        print(image)
        print('Average Test Accuracy: ' + str(acc))
        print('Average Test Loss: ' + str(loss))
        i+=1
    #acc = float(total/200)
    #avg_Loss = float(loss/200)
    return acc,loss


accuracy,loss = avg_accuracy()

accuracy,loss







'''
land=[0,0,0]
lakes=[39,39,244]
trees=[10,104,10]

COLOR_DICT = np.array([land,lakes,trees])
'''



