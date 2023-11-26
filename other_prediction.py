import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import random
import shutil
from convert_pdf_to_jpg import delete_files_in_directory



def copy_images(from_dir,to_dir,r=0.5):

    for image_type in os.listdir(from_dir):

        type_dir = os.path.join(from_dir, image_type)
        image_files = os.listdir(type_dir)

        if not os.path.exists(to_dir):
            os.makedirs(to_dir)

        num = 0

        for file in image_files:
            if random.random() > r:
                new_filename = f'{image_type}_{num}.jpg'  # Generate new filename
                new_path = os.path.join(to_dir, new_filename)  # New path with the new filename
                shutil.copy(os.path.join(type_dir, file), new_path)
                num += 1


def preprocess_new_data(file_path, img_height=224, img_width=224):

    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.array(img)
    return img_array

