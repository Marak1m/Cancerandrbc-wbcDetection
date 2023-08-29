import numpy as np
import torch
import glob
from IPython.display import Image, display
import os
from flask import Flask, flash, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import shutil

UPLOAD_FOLDER = 'D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\val'
OUTPUT_FOLDER = 'D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\output'
ALLOWED_EXTENSIONS = set(['jpg'])


app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', 'D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\best.pt')


@app.route('/')
def home():
    return render_template('index.html')

# take an image input to detect the objects in it
@app.route('/',methods=['POST'])

def upload_image_and_get_prediction():
    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files['file']
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        res = glob.glob('D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\val\\*.jpg')
        results = model(res, size=640)
        results.save(save_dir='D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\output')
        results.show()
        # get_prediction()
        folder_path = (r'D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\val')
        test = os.listdir(folder_path)
        for images in test:
            if images.endswith(".jpg"):
                os.remove(os.path.join(folder_path, images))
        dir_path = (r"D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\output")
        shutil.rmtree(dir_path, ignore_errors=True)
        return render_template('page.html')

# def get_prediction():
#         res = glob.glob('D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\val\\*.jpg')
#         results = model(res, size=640)
#         results.show()
        # os.remove('D:\\Bennett Courses\\SEM 3\\AI\\VSC\\deploy-ML-model-on-AWS-EC2-instance-main\\val\\*.jpg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    #app.run(debug=True)
    
    
    

    
