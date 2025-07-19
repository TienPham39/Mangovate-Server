from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

print("✅ Building clean model architecture...")

# Bước 1: Tạo lại kiến trúc từ đầu (KHÔNG dùng model cũ)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None, pooling='avg')

x = Dense(224, activation='relu')(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Bước 2: Load trọng số từ file .keras
print("🔁 Loading weights from .keras file...")
model.load_weights("best_mobilnetv2_model.keras")

# Bước 3: Save lại file .h5 sạch hoàn toàn
print("💾 Saving clean .h5 model...")
model.save("best_mobilnetv2_model.h5")
print("🎉 Done! Saved as best_mobilnetv2_model.h5")
