import tensorflow as tf
import numpy as np
import os
from app.server.model.utils.preprocess_images import preprocess_images

def predict_severity(image):
  current_dir = os.path.dirname(__file__)
  model_path = os.path.join(current_dir, 'saved_models', 'diabetic_retinopathy_model.h5')

  model = tf.keras.models.load_model(model_path)
  img, _ = preprocess_images([image])

  prediction = model.predict(img)

  print(f'Severity level: {np.argmax(prediction)}')
  return np.argmax(prediction)


# image = '179_right.jpeg' # level 0
# image = '179_left.jpeg' # level 0
# image = '217_left.jpeg' # level 4
# image = '217_right.jpeg' # level 4
# predict_severity(image)