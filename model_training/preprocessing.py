import pandas as pd
import os
import re
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

def load_and_preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array /= 255.0 
    return img_array

def find_image_filename(image_id, folder_path):
    numeric_id = re.search(r'\d+', image_id).group()  
    for filename in os.listdir(folder_path):
        if numeric_id in filename:
            return os.path.join(folder_path, filename)
    return None

def load_images_from_dataset(dataset, folder_path, target_size):
    image_data = []
    labels = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        image_id = row['image_id']
        lesion_type = row['lesion_type']
        image_path = find_image_filename(image_id, folder_path)

        if image_path is not None:
            image_data.append(load_and_preprocess_image(image_path, target_size))
            labels.append(lesion_type)

    return np.array(image_data), np.array(labels)

def balance_data(df):
    df_balanced = pd.DataFrame()
    
    for les in df['lesion_type'].unique():
        temp = resample(df[df['lesion_type'] == les], 
                        replace=True,     
                        n_samples=6000,   
                        random_state=123) 

        df_balanced = pd.concat([df_balanced, temp])

    return df_balanced
