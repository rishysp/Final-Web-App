from flask import Flask,render_template,request
import cv2
from keras.applications.densenet import DenseNet121
import numpy as np
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout
from keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os,random


model1 = tf.keras.applications.DenseNet201(input_shape=(150,150,3),include_top=False,weights='imagenet',pooling='avg')

''' freezing layers '''
model1.trainable = False

inp = model1.input
''' Hidden Layer '''
x = tf.keras.layers.Dense(128, activation='relu')(model1.output)
''' Classification Layer '''
out = tf.keras.layers.Dense(4, activation='softmax')(x)

''' Model '''
model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

model.load_weights('mine_model_weights.h5')


lbl_encoding = {0:'no_tumor', 1:'pituitary_tumor', 2:'meningioma_tumor', 3:'glioma_tumor'}
def map_label(val):
    return lbl_encoding[val]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after',methods = ['GET','POST'])
def after():
    global model

    
    file = request.files['file1']
    file.save('static/file.jpg')
    
    
    randomimage = load_img('static/file.jpg',target_size=(150,150))
    ''' converting img to array '''
    randomimage = img_to_array(randomimage) 

    ''' scaling '''
    img = randomimage/255.0

    ''' expanding dimensions '''
    img = np.expand_dims(img, axis=0)

    ''' predicion '''
    pred = model.predict(img)

    ''' retreiving max val from predited values'''
    val = np.argmax(pred)
    map_val  = map_label(val)

    
    return render_template('after.html',final = map_val)    

if __name__ == '__main__':
    app.run()