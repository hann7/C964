import os.path
import pandas as pd

sample_per_category_size = 40

try:
    file_path = '../data/trainLabels.csv'
    # check that the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)

    # check that we have enough samples in each severity level
    if df['level'].value_counts().min() < sample_per_category_size:
        raise ValueError(f"Not all severity levels have at least {sample_per_category_size} samples")

    def get_sample(group):
        # take 40 random images from each severity level
        return group.sample(sample_per_category_size, random_state=42)

    random_samples = df.groupby('level', group_keys=False).apply(get_sample)

    # save to new csv
    output_path = '../data/sample_200_labels.csv'
    random_samples.to_csv(output_path, index=False)

    print(f"Selected and saved {sample_per_category_size * 5} labels for data subset")

except Exception as e:
    print(f"Error: {e}")