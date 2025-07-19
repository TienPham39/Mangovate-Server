import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt tối ưu oneDNN để tránh sai lệch kết quả nhỏ

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw
from flask_cors import CORS

from io import BytesIO
import base64

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Tải model
MODEL_PATH = "best_mobilnetv2_model.keras"
print(f"Đang tải mô hình từ: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Đã tải mô hình thành công.")

# Danh sách nhãn tương ứng với các lớp mô hình
labels = ['Disease', 'Partially Ripe', 'Ripe', 'Unripe']

@app.route('/')
def home():
    return "Mango Classification API đang hoạt động. Truy cập /api/predict để sử dụng.", 200

@app.route('/api/predict', methods=['POST'])
def predict():
    print("Đã nhận yêu cầu /api/predict")

    if 'file' not in request.files:
        print("Không tìm thấy file trong request")
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400

    file = request.files['file']
    print(f"Đã nhận file: {file.filename}")

    try:
        # Xử lý ảnh
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print("Ảnh đã xử lý xong.")

        # Dự đoán
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        print(f"Dự đoán: {labels[class_index]} - Độ tin cậy: {confidence:.2f}")

        # Vẽ kết quả lên ảnh
        draw = ImageDraw.Draw(img)
        label_text = f"{labels[class_index]} ({confidence*100:.1f}%)"
        draw.rectangle([(0, 0), (img.width, 30)], fill=(255, 255, 255, 180))
        draw.text((10, 5), label_text, fill="black")

        # Chuyển ảnh sang base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'class': labels[class_index],
            'confidence': round(confidence, 4),
            'annotated_image': img_base64
        })

    except Exception as e:
        print(f"Lỗi xử lý ảnh: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"Chạy server tại cổng {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
