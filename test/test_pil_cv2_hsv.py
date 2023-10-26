
from PIL import Image

import numpy as np
import cv2


img_path = '/data/hdd1/by/tmp_folder/test.jpg'


img_cv2 = cv2.imread(img_path)
img_cv2_hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)

print(img_cv2_hsv[:, :, 2].max())
# print(img_cv2_hsv[:, :, 0].max())


img_pil = Image.open(img_path)
img_pil = img_pil.convert('HSV')

# H, S, V = img_pil.split()
img_pil = np.array(img_pil)
# print(img_pil[:, :, 0])
print(img_pil[:, :, 0].max())