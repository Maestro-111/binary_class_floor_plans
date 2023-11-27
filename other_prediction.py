import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import random
import shutil
from convert_pdf_to_jpg import delete_files_in_directory
from make_dataset import enh
from convert_pdf_to_jpg import make_floor_plans
from convert_pdf_to_jpg import copy_sur

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

def make_test_dir(r):
    make_floor_plans(pdf_dir = 'in_pdf',output_folder_path='test_data',prob=r)
    copy_sur(data_dir="survey_original", out_dir='test_data',prob=r)


def preprocess_new_data(file_path, factor, img_height=224, img_width=224):
    img = Image.open(file_path)
    img = enh(img, factor, (img_height,img_width))
    img_array = np.array(img)
    return img_array

