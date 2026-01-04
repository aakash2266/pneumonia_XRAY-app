from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import os

app = Flask(__name__)

# Load model
model = keras.models.load_model('pneumonia_model (1).keras')

# Compile it
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        filepath = os.path.join('static/uploads', file.filename)
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
            confidence = prediction * 100
        else:
            result = "Normal"
            confidence = (1 - prediction) * 100
        
        os.remove(filepath)
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2f}%"
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)