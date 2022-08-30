import cv2
import os
from PIL import Image

def resize_fct(folder):
    s_h = 0
    s_w = 0
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        # print("img type:", type(filename))
        if img is not None:
            img = img.resize((357, 325))
            img.save(filename)

h, w = resize_fct('./')