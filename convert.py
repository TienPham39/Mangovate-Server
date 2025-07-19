from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

# Xây lại mô hình đúng như lúc train (đảm bảo không có batch_shape)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None, pooling='avg')

x = Dense(224, activation='relu')(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Load trọng số từ model đã train (phải là file .keras)
model.load_weights("best_mobilnetv2_model.keras")

# Lưu lại HDF5 chuẩn – không còn batch_shape!
model.save("best_mobilnetv2_model.h5")
print("Saved final clean .h5 model")
