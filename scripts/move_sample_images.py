import os
import shutil
import pandas as pd

#  Script to move the gathered data subset to a new folder to work with

images_path = '../dataset/train/train'
output_path = '../data/sample_dataset_raw'

# create folder if it does not exist
os.makedirs(output_path, exist_ok=True)

image_labels = '../data/sample_labels.csv'
df = pd.read_csv(image_labels)

for row in df.itertuples():
    image_filename = row.image + '.jpeg'
    image_path = os.path.join(images_path, image_filename)

    if os.path.exists(image_path):
        try:
            shutil.copy(image_path, os.path.join(output_path, image_filename))
        except Exception as e:
            print(f'Error copying image: {image_filename}: {e}')
    else:
        print(f'Image not found: {image_filename}')

print(f'Images copied!!')