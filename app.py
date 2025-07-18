from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS

from io import BytesIO
import base64

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi từ domain khác (localhost:3000, v.v.)

# Load mô hình MobileNetV2
model = tf.keras.models.load_model('best_mobilnetv2_model.keras')

# Danh sách nhãn tương ứng với các lớp mô hình
labels = ['Disease', 'Partially Ripe', 'Ripe', 'Unripe']

@app.route('/api/predict', methods=['POST'])  
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400

    file = request.files['file']

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])

        # === Vẽ nhãn lên ảnh ===
        draw = ImageDraw.Draw(img)
        label_text = f"{labels[class_index]} ({confidence*100:.1f}%)"
        draw.rectangle([(0, 0), (img.width, 30)], fill=(255, 255, 255, 180))
        draw.text((10, 5), label_text, fill="black")

        # Convert ảnh sang base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'class': labels[class_index],
            'confidence': round(confidence, 4),
            'annotated_image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8080)
