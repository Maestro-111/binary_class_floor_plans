import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance



# Read the RGB image
image = Image.open('C:/real_copy_rep/floorplan/src/classifier/survey_original/images (49).jpeg')


def check_rgb_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read the image.")
        return

    if len(image.shape) < 3 or image.shape[2] < 3:
        print("The image is not in RGB format.")
    else:
        print("The image is in RGB format.")

# Replace 'path_to_your_image.jpg' with the actual path to your image file

check_rgb_image('C:/real_copy_rep/floorplan/src/classifier/Dataset_original/train/surveys/aug_0_254.jpg')


exit()

enhancer = ImageEnhance.Sharpness(image)
image = image.resize((224,224), Image.ANTIALIAS)
sharpened_img = enhancer.enhance(5)

image.save('s1.jpeg')

image = cv2.imread("s1.jpeg")

image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(10, 10))
final_img = clahe.apply(image_bw)

# Original RGB image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Edge-detected image using Sobel
plt.subplot(1, 2, 2)
plt.imshow(final_img, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()



rgb_image = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
cv2.imwrite("s1.jpeg", rgb_image)


