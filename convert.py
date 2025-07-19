from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

try:
    print("Loading .keras model...")
    original_model = load_model("best_mobilnetv2_model.keras", compile=False)

    print("Rebuilding model without batch_shape...")
    x = Input(shape=(224, 224, 3))
    y = original_model(x)
    final_model = Model(inputs=x, outputs=y)

    print("Saving new clean .h5 model...")
    final_model.save("best_mobilnetv2_model.h5")
    print("Done! Saved as 'best_mobilnetv2_model.h5'")

except Exception as e:
    print("Error during conversion:", e)
