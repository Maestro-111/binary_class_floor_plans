import os
from PIL.Image import Image
import fitz


def removeFile(filename):
    if os.path.exists(filename):
        os.remove(filename)

def extract_images_from_pdf(pdf_path, output_folder):
    imageFiles = []

    pdf_filename = os.path.basename(pdf_path)
    pdf_doc = fitz.open(pdf_path)

    for page_index in range(len(pdf_doc)):

        filepath = f'{output_folder}/pdf_{pdf_filename}_{page_index}.jpg'

        print(filepath)

        # get the page itself
        page = pdf_doc[page_index]
        # image_list = page.getImageList()

        pix = page.get_pixmap(matrix=fitz.Identity, dpi=250)

        #print(pix)
        removeFile(filepath)
        pix.save(filepath)  # save file
        imageFiles.append(filepath)
    return imageFiles

def make_floor_plans(pdf_dir = 'in_pdf',output_folder_path = "floor_palns"):

    for pdf_file in os.listdir(pdf_dir):
        path = os.path.join(pdf_dir, pdf_file)
        extract_images_from_pdf(path, output_folder_path)


def delete_files_in_directory(directory_path):
    try:
        # List all files in the specified directory
        files = os.listdir(directory_path)

        # Iterate through each file and delete them
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("All files deleted successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

