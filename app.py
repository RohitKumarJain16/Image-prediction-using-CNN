import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

try:
    model = load_model('cat_dog_model.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = None
    image_path = None
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
            
        if not file or not file.filename:
            return redirect(request.url)
        if not allowed_file(file.filename):
            return redirect(request.url)
        filename = secure_filename(file.filename)
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
       
        if model:
            try:
               
                img = Image.open(image_path).resize((150, 150))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0) 
                prediction = model.predict(img_array)
                if prediction[0][0] > 0.5:
                    prediction_result = "It's a Dog! 🐶"
                else:
                    prediction_result = "It's a Cat! 🐱"
            except Exception as e:
                prediction_result = f"Error during prediction: {e}"
        else:
            prediction_result = "Error: Model not loaded."
                
    return render_template('prediction.html', prediction_result=prediction_result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
