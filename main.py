from flask import Flask, render_template
from flask import request
from flask import redirect, url_for
import os
import pickle
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from sklearn.preprocessing import LabelBinarizer

app = Flask(__name__)


BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

## -------------------- Load Models -------------------

model = tf.keras.models.load_model('static/models/scene_classification.model')

scene_cls_path = os.path.join(MODEL_PATH,'scene_classification_lb.pickle')
lb = pickle.loads(open(scene_cls_path, 'rb').read())
#model_path = os.path.join(MODEL_PATH,'scene_classification.model')



#model_sgd = pickle.load(open(model_sgd_path,'rb'))
#scaler = pickle.load(open(scaler_path,'rb'))



@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 OCCURED. Page Not Found. Please go the home page and try again"
    return render_template("error.html",message=message) # page not found

@app.errorhandler(405)
def error405(error):
    message = 'Error 405, Method Not Found'
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message='INTERNAL ERROR 500, Error occurs in the program'
    return render_template("error.html",message=message)


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename 
        print('The filename that has been uploaded =',filename)
        # know the extension of filename
        # all only .jpg, .png, .jpeg, PNG
        ext = filename.split('.')[-1]
        print('The extension of the filename =',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            # saving the image
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved sucessfully')
            # send to pipeline model
            results = pipeline_model(path_save)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results,image_filename=filename,height=hei)


        else:
            print('Use only the extension with .jpg, .png, .jpeg')

            return render_template('upload.html',extension=True,fileupload=False)
            
    else:
        return render_template('upload.html',fileupload=False,extension=False)

@app.route('/about/')
def about():
    return render_template('about.html')

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ =img.shape 
    aspect = h/w
    given_width = 300
    height = given_width*aspect
    return height

def pipeline_model(path):
    print('path',path)
    # pipeline model
    # load the test image
    image = cv2.imread(path)
    output = image.copy()
    image = cv2.resize(image, (128, 128))

    # scale the pixels
    image = image.astype('float') / 255.0

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    print(image)
    # predict
    # predict
    
    # predict
    preds = model.predict(image)

    # get the class label
    max_label = preds.argmax(axis=1)[0]
    print('PREDICTIONS: \n', preds)
    print('PREDICTION ARGMAX: ', max_label)
    label = lb.classes_[max_label]
    print('labels****',label)

    text = '{}: {:.2f}%'.format(label, preds[0][max_label] * 100)
    top_dict = dict()
    top_dict={label: preds[0][max_label] * 100}
    print('top_dict***',top_dict)
    return top_dict
    

if __name__ == "__main__":
    app.run(debug=False) 
