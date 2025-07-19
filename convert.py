from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

# Tạo lại kiến trúc gốc
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None,  # Không dùng imagenet vì bạn sẽ load trọng số riêng
    pooling='avg'
)

x = Dense(224, activation='relu')(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Load trọng số từ file đã train
model.load_weights("best_mobilnetv2_model.keras")

# Save lại HDF5 chuẩn, KHÔNG có batch_shape
model.save("best_mobilnetv2_model.h5")
print("Final clean .h5 model saved.")
