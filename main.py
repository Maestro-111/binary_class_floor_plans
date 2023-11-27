
import os
import re

from sample import create_datasets
from sample import make_confusion_matrix
from sample import neural_net_mixin
from sample import CNN

from other_prediction import copy_images
from convert_pdf_to_jpg import delete_files_in_directory
from other_prediction import preprocess_new_data
from other_prediction import make_test_dir

from convert_pdf_to_jpg import delete_files_in_directory
from convert_pdf_to_jpg import make_floor_plans
from convert_pdf_to_jpg import extract_images_from_pdf
from convert_pdf_to_jpg import removeFile
from convert_pdf_to_jpg import copy_sur

from make_dataset import augementation
from make_dataset import count_files_in_directory
from make_dataset import save_to_dataset
from make_dataset import enh
from make_dataset import sharp_and_res

from delete_all import clear_dirs

import numpy as np
import tensorflow as tf
from tensorflow import keras

FACTOR = 5


def preprocess(factor):

    delete_files_in_directory(directory_path='data/floor_plans')
    delete_files_in_directory(directory_path='data/surveys')


    make_floor_plans(output_folder_path='data/floor_plans')
    copy_sur(data_dir="survey_original", out_dir='data/surveys')

    print("Converted PDFS")


    for type_dir in ['train', 'test', 'validation']:
        for type_image in ['floor_plans', 'surveys']:
            path = f'Dataset_original/{type_dir}/{type_image}'
            delete_files_in_directory(directory_path=path)

    sharp_and_res(factor=factor)

    print("Transformed Images")

    save_to_dataset()

    print("Made dataset")

    augementation('Dataset_original/train/surveys',
                  'Dataset_original/train/floor_plans', 'surveys')
    augementation('Dataset_original/test/surveys',
                  'Dataset_original/test/floor_plans', 'surveys')
    augementation('Dataset_original/validation/surveys',
                  'Dataset_original/validation/floor_plans', 'surveys')

    print('Done augementation')

def create_data(train_path,val_path,test_path):
    train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_datasets(train_path,val_path,test_path)

    print(class_names)
    print(num_classes)

    return train_dataset, validation_dataset, test_dataset, class_names, num_classes




def train_model_and_save(train_dataset, validation_dataset, test_dataset,num_classes):


    CNN_net = CNN(num_classes)
    history = CNN_net.train(train_dataset,validation_dataset,epochs=10)
    CNN_net.plot_training_hist(history, '3-layers CNN', ['red', 'orange'], ['blue', 'green'])
    CNN_net.evaluate_model(test_dataset,class_names)
    CNN_net.save('model.h5')


def make_preds(class_names,model_path,factor,prob):
    loaded_model = keras.models.load_model(model_path)

    print(loaded_model)

    delete_files_in_directory('test_data')
    make_test_dir(r=prob)


    data_directory = 'test_data'

    file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]

    # Make predictions
    predictions = []
    for file_path in file_paths:
        img = preprocess_new_data(file_path,factor, 224,224)
        img = tf.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction using the loaded model
        prediction = loaded_model.predict(img)

        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        predictions.append(predicted_class)

    # Display predictions (you might adjust this based on your needs)

    overall = 0
    correct = 0

    for file_path, prediction in zip(file_paths, predictions):
        print(file_path)
        if re.findall(r'[Ff]loor\s?_?\+?\-?[Pp]lan[s]?',file_path):
            label = 'floor_plans'
        else:
            label= 'surveys'

        if label == prediction:
            correct += 1
        print(f"File: {file_path}, Label: {label} , Prediction: {prediction}")
        overall += 1

    print(correct / overall)
    print(f'Accuracy: {round(correct / overall, 3)}')

make_prep = False
train_save= False
d = False

if __name__ == "__main__":

    if make_prep:
        preprocess(factor=FACTOR)

    train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_data("Dataset_original/train",
                "Dataset_original/validation",
                "Dataset_original/test")

    if train_save:
        train_model_and_save(train_dataset, validation_dataset, test_dataset,num_classes)

    make_preds(class_names,model_path='model.h5',factor=FACTOR,prob=0.4)

    if d:
        clear_dirs()