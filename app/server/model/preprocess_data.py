import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from app.server.model.utils import preprocess_images

def process_and_split_data():
    labels_file_path = '../../../data/sample_labels.csv'
    images_path = '../../../data/sample_dataset_processed'

    labels_df = pd.read_csv(labels_file_path)

    images = []
    labels = []

    for row in labels_df.itertuples():
        img_name = row.image + '.jpeg'
        level = row.level

        full_path = os.path.join(images_path, img_name)
        images.append(full_path)
        labels.append(level)


    processed_images, processed_labels = preprocess_images(images, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        processed_images,
        processed_labels,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test