from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(image_path):
    # Load and preprocess the image for your model
    img = Image.open(image_path)
    img = img.resize((256, 256))  # adjust the size according to your model's input
    img_array = np.array(img)  # normalize pixel values
    img_array = img_array.astype(np.float32) / 255.0  # Convert to float and normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

def predict_image_class(image_array, model, class_names):
    batch_prediction = model(image_array)
    predicted_class_index = np.argmax(batch_prediction[0])
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # dropdown_data = request.form['dropdown_data']

        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction='No selected file')

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Display image in Flask application web page
            uploaded_image = file.filename

            # Load the saved TensorFlow model
            saved_model_dir = ('/Users/sumlipuri/Desktop/plant_disease_detection-main/plant_village_denseNet.h5')
            model = load_model(saved_model_dir)

            dataset = tf.keras.preprocessing.image_dataset_from_directory('/Users/sumlipuri/Desktop/plant_disease_detection-main/DataSet/color')
            class_names = dataset.class_names

            # Process the uploaded image
            img_array = process_image(filename)

            predicted_class_name = predict_image_class(img_array, model, class_names)

            # You can continue with the rest of your code for generating remedies and symptoms here

            return render_template('index.html', prediction=predicted_class_name, uploaded_image=uploaded_image)

    return render_template('index.html', prediction=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)