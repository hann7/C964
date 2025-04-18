from PIL import Image
import os

input_folder = '../data/sample_dataset_raw'
output_folder = '../data/sample_dataset_processed'

# create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

target_size = (224, 224)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpeg'):
        try:
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size)

            image.save(os.path.join(output_folder, filename))
        except Exception as e:
            print(f'Error processing {filename}: {e}')

print('Images have been resized')