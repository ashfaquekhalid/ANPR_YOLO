import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np
from anpr import plate_number
#from flask_ngrok import run_with_ngrok
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#run_with_ngrok(app)

@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = plate_number(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run()
