import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import json
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0 
    return img_array

def classify_images(model_path, image_dir, target_size, output_json):
    model = load_model(model_path)
    results = []
    lesion_type_mapping = {
        
        0: "BKL",
        1: "NV",
        2: "DF",
        3: "MEL",
        4: "VASC",
        5: "BCC",
        6: "AKIEC"
    }

    test_image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
    test_images = [preprocess_image(image_path, target_size) for image_path in tqdm(test_image_paths)]
    test_images = np.vstack(test_images)
    predictions = model.predict(test_images)

    
    for image_path, prediction in zip(test_image_paths, predictions):
        predicted_label = lesion_type_mapping[np.argmax(prediction)]
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        results.append({"image_id": img_id, "lesion_type": predicted_label})

    with open(output_json, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    model_path = '/Users/donika/Desktop/images/model_training/model.h5'  
    image_dir = '/Users/donika/Desktop/images/datasets/test'  
    target_size = (128, 128)
    output_json = 'JSON.json'

    classify_images(model_path, image_dir, target_size, output_json)

