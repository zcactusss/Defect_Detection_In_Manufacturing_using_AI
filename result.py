import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved brain
model = tf.keras.models.load_model('defect_model.h5')

def classify_part(img_path):
    if not os.path.exists(img_path):
        return "File not found!"
    
    # Prep the image
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    result = model.predict(img_array, verbose=0)
    
    # Output logic based on our class indices (0=Defective, 1=OK)
    if result[0] < 0.5:
        confidence = (1 - result[0][0]) * 100
        return f"❌ REJECTED: Defect Detected ({confidence:.2f}% confidence)"
    else:
        confidence = result[0][0] * 100
        return f"✅ PASSED: Quality OK ({confidence:.2f}% confidence)"

# TEST IT
test_file = r"D:\Beta\model\defect1.jpeg"# Update this path
print(f"Analyzing {test_file}...")
print(classify_part(test_file))
