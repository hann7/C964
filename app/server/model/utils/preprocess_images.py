from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

def preprocess_images(image_paths, labels=None, target_size=(224, 224)):
    images = []
    processed_labels = [] if labels is not None else None

    for i, path in enumerate(image_paths):
        try:
            
            if isinstance(path, str):
              img = image.load_img(path, target_size=target_size)
            elif hasattr(path, 'size'):
              img = path.resize(target_size)
            else:
              raise ValueError(f"Invalid input type: {type(path)}. Expected string (file path) or PIL Image.")

            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)

            if labels is not None:
              processed_labels.append(labels[i])

        except Exception as e:
            print(f'Failed to process file: {path}: {e}')

    images = np.array(images)

    if labels is not None:
        processed_labels = np.array(processed_labels)
        return images, processed_labels

    return images, None