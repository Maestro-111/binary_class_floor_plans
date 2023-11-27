import os
from convert_pdf_to_jpg import delete_files_in_directory


def clear_dirs():
    delete_files_in_directory(directory_path='data/floor_plans')
    delete_files_in_directory(directory_path='data/surveys')

    for type_dir in ['train', 'test', 'validation']:
        for type_image in ['floor_plans', 'surveys']:
            path = f'Dataset_original/{type_dir}/{type_image}'
            delete_files_in_directory(directory_path=path)

    delete_files_in_directory('some_data')
    delete_files_in_directory('test_data')

clear_dirs()