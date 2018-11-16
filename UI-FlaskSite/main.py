from flask import Flask, request, render_template, send_from_directory
import os
from binascii import a2b_base64
import random
import numpy as np
from matplotlib import pylab as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import model_from_yaml
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
from resizeimage import resizeimage
import tensorflow as tf

app = Flask(__name__)

#Change Path for Configuration
working_dir = os.path.join("/./","")
app.config['UPLOAD_FOLDER'] = '/./uploads'

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(os.path.join(working_dir,"uploads"), filename, as_attachment=False)

@app.route('/')
def index():
    imghidden = 'hidden'
    canhidden = ''
    rint = random.randint(0,50000)
    return(render_template('home.html',**locals()))

@app.route('/upload', methods=['POST'])
def upload():
    img_data = request.form['img1']
    img_data = img_data[22:]
    binary_data = a2b_base64(img_data)
    fd = open(os.path.join(working_dir,"uploads/image1.png"), 'wb')
    fd.write(binary_data)
    fd.close()
    return(process())

@app.route('/upload-manual', methods=['POST'])
def uploadManual():
    #Save image
    try:
        image = request.files['img2']
        print(image.filename)
    except:
        return(index())
    #Convert image to png for use with process
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    print(os.path.join(working_dir,"uploads/image1.png"))
    rgb_im.save(os.path.join(working_dir,"uploads/image1.png"))

    with open(os.path.join(working_dir,"uploads/image1.png"), 'rb') as f:
                with Image.open(f) as image:
                    try:
                        cover = resizeimage.resize_cover(image, [320, 240])
                        cover.save(os.path.join(working_dir,"uploads/image1.png"), image.format)
                    except:
                        print('Error on ')

    return(process())

@app.route('/process')
def process():
    im = Image.open(os.path.join(working_dir,"uploads/image1.png"))
    rgb_im = im.convert('RGB')
    rgb_im.save(os.path.join(working_dir,"uploads/image1.jpg"))

    with open(os.path.join(working_dir,"uploads/image1.jpg"), 'rb') as f:
                with Image.open(f) as image:
                    try:
                        cover = resizeimage.resize_cover(image, [28, 28])
                        cover.save(os.path.join(working_dir,"uploads/image1.jpg"), image.format)
                    except:
                        print('Error on ')

    img_path = os.path.join(working_dir,"uploads/image1.jpg")
    # Convert to Numpy Array
    img = Image.open(img_path)
    img.load()
    test_image = np.asarray( img, dtype="float32" )
    test_image = test_image[np.newaxis, ...]

    imghidden = ''
    canhidden = 'hidden'
    rint = random.randint(0,50000)

    sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=1))

    with sess:
        # Running the model_yaml and predicting numbers
        with open(os.path.join(working_dir,'model/CDModel-BNB-SGD.yaml'), 'r') as yaml_file:
            loaded_yaml_model = yaml_file.read()

        # Create model from yaml config and load weights fron trained networks
        loaded_model = model_from_yaml(loaded_yaml_model)
        loaded_model.load_weights(os.path.join(working_dir,'model/CDModel-BNB-SGD.h5'))
        print("Loaded model and weights")

        #Create Prediction
        prediction = loaded_model.predict(test_image, batch_size=70, verbose=1)
        if np.argmax(prediction) == 0:
            item = 'Bottle'
        else:
            item = 'Not Bottle'

        message = 'Prediction: {} with {:.2f}% confidence.'.format(item, np.max(prediction)*100)
        print(message)

    return(render_template('home.html',**locals()))
