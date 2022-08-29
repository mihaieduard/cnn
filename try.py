from PIL import Image
import os

def load_images_from_folder(folder):
    i = 0
    s_h = 0
    s_w = 0
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        # print("img type:", type(filename))
        if img is not None:
            i += 1
            w,h = img.size
            print(w, h)
        if i == 2:
            break

load_images_from_folder('cal')