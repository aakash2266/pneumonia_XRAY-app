from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model = keras.models.load_model('pneumonia_model.keras')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        if prediction > 0.5:
            result = "Pneumonia Detected"
            result_class = "danger"
            confidence = prediction * 100
            recommendation = "Please consult a healthcare professional immediately for proper diagnosis and treatment."
        else:
            result = "Normal"
            result_class = "success"
            confidence = (1 - prediction) * 100
            recommendation = "Your chest X-ray appears normal. However, always consult with a medical professional for accurate diagnosis."
        
        # Keep the image for display
        image_url = url_for('static', filename=f'uploads/{file.filename}')
        
        return jsonify({
            'result': result,
            'result_class': result_class,
            'confidence': f"{confidence:.2f}%",
            'recommendation': recommendation,
            'image_url': image_url
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error analyzing image: {str(e)}'})

if __name__ == "__main__":
    import os
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)