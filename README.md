# Seneca Floorplan Project
this repo is for RM floor project.

---
- Phase 1 progress 95%
- Phase 2 progress 0
- Phase 3 progress 20%
- Phase 4 progress 0
- Phase 5 progress 0

## Binary (Multi Class) Classifier
- just run the main.py with the corresponding param: pass either 0 or 1 for 'predictions' and for 'training'
- be sure to establish env var to image dir data. We use 3 for training, one for floorplans, one for other class and one for the surveys, and same 3 for predictions.

env names sample for traning and for predictions (env var are named like this on my side)

floorplan_image_path
survey_image_path
other_image_path



# Contribute
* install poetry, see https://python-poetry.org/docs/
* install dependencies by ```poetry install```

## Floorplan processing (phase 3)

- extract address and project name to DB and/or folders
- read source folder:
  - read target info.json to get previous info
    - establish initial page index
  - split into single page of images if pdf file
  - process each image as a page if it's an image file
- split each page to
  - floorplan: biggest image. Feed for future ML
  - keyplate: may have multiple. Feed for future ML
  - unit name
  - sqft (+balcany sqft)
  - floors
  - remarks
  - disclaimers
- save all info to DB and/or folders


### Database structure (mongo)
TODO:

# setup enviroment
```
poetry install
pip install keras-ocr
pip install pytesseract
pip install opencv-python
pip install easyocr
pip install pdf2image
pip install imagehash
pip install PyMuPDF
pip install fitz
```

# Copyright/License/Disclamer
apache-2.0 license.
TODO:
