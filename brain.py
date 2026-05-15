import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model('defect_model.h5')

def predict(img_path):

    if not os.path.exists(img_path):
        return "File not found!"

    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)

    if prediction[0][0] < 0.5:
        return "❌ DEFECTIVE PRODUCT"
    else:
        return "✅ NON-DEFECTIVE PRODUCT"


# TEST IMAGE
test_image = r"dataset/test/defective/defect1.jpeg"

print(predict(test_image))
