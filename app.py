import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt tối ưu oneDNN để tránh sai lệch kết quả nhỏ

import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

from io import BytesIO
import base64

# Khởi tạo Flask app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mangovate-fe.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("best_mobilnetv2_model.keras")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy mô hình tại {MODEL_PATH}")

labels_map = {0: "Disease", 1: "Partially Ripe", 2: "Ripe", 3: "Unripe"}

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predicted/")
async def analyze(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        processed = preprocess_image(img)

        tf.keras.backend.clear_session()
        model = load_model(str(MODEL_PATH), compile=False)

        prediction = model.predict(processed)[0]
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        predicted_label = labels_map[class_index]

        # Annotate ảnh
        # draw = ImageDraw.Draw(img)
        # label_text = f"{predicted_label} ({confidence:.2f}%)"
        # draw.rectangle([(0, 0), (img.width, 30)], fill=(255, 255, 255, 180))
        # draw.text((10, 5), label_text, fill="black")

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Tạo dict chứa phần trăm từng lớp
        class_confidences = {
            labels_map[i]: round(float(prediction[i]) * 100, 2) for i in range(len(prediction))
        }

        return {
            "predicted_class": predicted_label,
            "confidence": round(confidence, 2),
            "annotated_image": img_base64,
            "all_confidences": class_confidences
        }

    except Exception as e:
        return {"error": str(e)}

# @app.route('/')
# def home():
#     return "Mango Classification API đang hoạt động. Truy cập /api/predict để sử dụng.", 200

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     print("Đã nhận yêu cầu /api/predict")

#     if 'file' not in request.files:
#         print("Không tìm thấy file trong request")
#         return jsonify({'error': 'Không tìm thấy file ảnh'}), 400

#     file = request.files['file']
#     print(f"Đã nhận file: {file.filename}")

#     try:
#         # Xử lý ảnh
#         img = Image.open(file.stream).convert("RGB")
#         img = img.resize((224, 224))
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         print("Ảnh đã xử lý xong.")

#         # Dự đoán
#         predictions = model.predict(img_array)
#         class_index = np.argmax(predictions[0])
#         confidence = float(predictions[0][class_index])
#         print(f"Dự đoán: {labels[class_index]} - Độ tin cậy: {confidence:.2f}")

#         # Vẽ kết quả lên ảnh
#         draw = ImageDraw.Draw(img)
#         label_text = f"{labels[class_index]} ({confidence*100:.1f}%)"
#         draw.rectangle([(0, 0), (img.width, 30)], fill=(255, 255, 255, 180))
#         draw.text((10, 5), label_text, fill="black")

#         # Chuyển ảnh sang base64
#         buffered = BytesIO()
#         img.save(buffered, format="PNG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode()

#         return jsonify({
#             'class': labels[class_index],
#             'confidence': round(confidence, 4),
#             'annotated_image': img_base64
#         })

#     except Exception as e:
#         print(f"Lỗi xử lý ảnh: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 10000))
#     print(f"Chạy server tại cổng {port}")
#     app.run(debug=True, host='0.0.0.0', port=port)
