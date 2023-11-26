
import os

from sample import create_datasets
from sample import make_confusion_matrix
from sample import neural_net_mixin
from sample import CNN

from other_prediction import copy_images
from convert_pdf_to_jpg import delete_files_in_directory
from other_prediction import preprocess_new_data

from convert_pdf_to_jpg import delete_files_in_directory
from convert_pdf_to_jpg import make_floor_plans
from convert_pdf_to_jpg import extract_images_from_pdf
from convert_pdf_to_jpg import removeFile

from make_dataset import augementation
from make_dataset import count_files_in_directory
from make_dataset import save_to_dataset
from make_dataset import enh

import numpy as np
import tensorflow as tf
from tensorflow import keras


def preprocess():

    delete_files_in_directory(directory_path='data/floor_palns')
    make_floor_plans(output_folder_path='data/floor_palns')


    for type_dir in ['train', 'test', 'validation']:
        for type_image in ['floor_palns', 'surveys']:
            path = f'Dataset_original/{type_dir}/{type_image}'
            delete_files_in_directory(directory_path=path)

    save_to_dataset()

    augementation('Dataset_original/train/surveys',
                  'Dataset_original/train/floor_palns', 'surveys')
    augementation('Dataset_original/test/surveys',
                  'Dataset_original/test/floor_palns', 'surveys')
    augementation('Dataset_original/validation/surveys',
                  'Dataset_original/validation/floor_palns', 'surveys')






def train_model_and_save(train_path,val_path,test_path):

    train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_datasets(train_path,val_path,test_path)

    print(class_names)
    print(num_classes)


    CNN_net = CNN(num_classes)
    history = CNN_net.train(train_dataset,validation_dataset,epochs=10)
    CNN_net.plot_training_hist(history, '3-layers CNN', ['red', 'orange'], ['blue', 'green'])
    CNN_net.evaluate_model(test_dataset,class_names)
    CNN_net.save('model.h5')

    return class_names


def make_preds(class_names,model_path):
    loaded_model = keras.models.load_model(model_path)
    print(loaded_model)

    delete_files_in_directory('some_data')
    copy_images('data', 'some_data', r=0)

    new_data_directory = 'some_data'  # Replace with the actual path
    file_paths = [os.path.join(new_data_directory, f) for f in os.listdir(new_data_directory)]

    # Make predictions
    predictions = []
    for file_path in file_paths:
        # Load and preprocess each image
        img = preprocess_new_data(file_path)
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
        label = 'floor_palns' if 'floor' in file_path else 'surveys'
        if label == prediction:
            correct += 1
        print(f"File: {file_path}, Label: {label} , Prediction: {prediction}")
        overall += 1

    print(correct / overall)
    print(f'Accuracy: {round(correct / overall, 3)}')

make_prep = True


if __name__ == "__main__":

    if make_prep:
        preprocess()

    class_names = train_model_and_save("Dataset_original/train",
            "Dataset_original/validation",
            "Dataset_original/test")
    make_preds(class_names,model_path='model.h5')