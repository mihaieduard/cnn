import cv2
import os
from PIL import Image

def load_images_from_folder(folder):
    i = 0
    s_h = 0
    s_w = 0
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        # print("img type:", type(filename))
        if img is not None:
            i += 1
            h ,w = img.size
            # print(h, w)
            s_h = s_h + h
            s_w = s_w + w
        if i == 1800:
            break
    return s_h//1800, s_w//1800

def resize_fct(folder):
    i = 0
    s_h = 0
    s_w = 0
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        # print("img type:", type(filename))
        if img is not None:
            img = img.resize((357, 325))
            img.save(filename)

caine_h, caine_w = load_images_from_folder('caine')
print(caine_h, caine_w)

cal_h, cal_w = load_images_from_folder('cal')
print(cal_h, cal_w)

elefant_h, elefant_w = load_images_from_folder('elefant')
print(elefant_h, elefant_w )

fluture_h, fluture_w = load_images_from_folder('fluture')
print(fluture_h, fluture_w)

gaina_h, gaina_w = load_images_from_folder('gaina')
print(gaina_h, gaina_w)

oaie_h, oaie_w = load_images_from_folder('oaie')
print(oaie_h, oaie_w)

paiangen_h, paiangen_w = load_images_from_folder('paiangen')
print(paiangen_h, paiangen_w)

pisica_h, pisica_w = load_images_from_folder('pisica')
print(pisica_h, pisica_w)

vaca_h, vaca_w = load_images_from_folder('vaca')
print(vaca_h, vaca_w)

veverita_h, veverita_w = load_images_from_folder('veverita')
print(veverita_h, veverita_w)

print((caine_h + cal_h + elefant_h + fluture_h + gaina_h + oaie_h + paiangen_h + pisica_h + vaca_h + veverita_h)//10, (caine_w + caine_w + elefant_w + fluture_w + gaina_w + oaie_w + paiangen_w + pisica_w + vaca_w + veverita_w)//10 )
print((caine_h + cal_h + elefant_h + fluture_h + gaina_h + oaie_h + paiangen_h + pisica_h + vaca_h + veverita_h)/10, (caine_w + caine_w + elefant_w + fluture_w + gaina_w + oaie_w + paiangen_w + pisica_w + vaca_w + veverita_w)/10 )



caine_h, caine_w = resize_fct('caine')
# print(caine_h, caine_w)

cal_h, cal_w = resize_fct('cal')
# print(cal_h, cal_w)

elefant_h, elefant_w = resize_fct('elefant')
# print(elefant_h, elefant_w )

fluture_h, fluture_w = resize_fct('fluture')
# print(fluture_h, fluture_w)

gaina_h, gaina_w = resize_fct('gaina')
# print(gaina_h, gaina_w)

oaie_h, oaie_w = resize_fct('oaie')
# print(oaie_h, oaie_w)

paiangen_h, paiangen_w = resize_fct('paiangen')
# print(paiangen_h, paiangen_w)

pisica_h, pisica_w = resize_fct('pisica')
# print(pisica_h, pisica_w)

vaca_h, vaca_w = resize_fct('vaca')
# print(vaca_h, vaca_w)

veverita_h, veverita_w = resize_fct('veverita')
# print(veverita_h, veverita_w)