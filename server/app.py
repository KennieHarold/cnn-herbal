import os
from flask import Flask, flash, jsonify, request, redirect
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
import numpy as np
from werkzeug.utils import secure_filename

IMG_SIZE = 150
LR = 1e-3
MODEL = './models/herbal-0.001-6conv-basic.model'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# App
app = Flask(__name__) # Initialize the flask App

if not os.path.exists('temp'):
    os.makedirs('temp')
    
def load_model():
    convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input')
    
    convnet = conv_2d(convnet, 32, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation ='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 5, activation ='softmax')
    convnet = regression(convnet, optimizer ='adam', learning_rate = LR, loss ='categorical_crossentropy', name ='targets')

    model = tflearn.DNN(convnet, tensorboard_dir ='log')
    model.load(MODEL)
    
    return model

def convert_image(req_img):
    img = cv2.imread(req_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data = np.array(img)
    data = data.reshape(IMG_SIZE, IMG_SIZE, 1)
    return data

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return jsonify(message="Hello world!")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ext = filename.split(".")[1]
        test_file = "test." + ext
        file.save(os.path.join('temp', test_file))
        
        data = convert_image("temp/" + test_file)
        model = load_model()
        model_out = model.predict([data])[0]
        
        if np.argmax(model_out) == 0: 
            return jsonify(herbal='lagundi')
        elif np.argmax(model_out) == 1: 
            return jsonify(herbal='mayana')
        elif np.argmax(model_out) == 2: 
            return jsonify(herbal='oregano')
        elif np.argmax(model_out) == 3: 
            return jsonify(herbal='sambong')
        elif np.argmax(model_out) == 4: 
            return jsonify(herbal='yerba')