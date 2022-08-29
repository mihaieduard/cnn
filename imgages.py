from tkinter import N
import cv2
import os
import time 

def load_images_from_folder(folder):
    i = 0
    s_h = 0
    s_w = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        # print("img type:", type(filename))
        if img is not None:
            i += 1
            h ,w = img.shape
            print(h, w)
            s_h = s_h + h
            s_w = s_w + w
        # if i == 1800:
        #     break
    print(i)
    time.sleep(5)
    return s_h//i, s_w//i

time.sleep(1)
print("\n\n ---------- Caine ----------")
time.sleep(1)
caine_h, caine_w = load_images_from_folder('caine')
print(caine_h, caine_w)

time.sleep(1)
print("\n\n ---------- Cal ----------")
time.sleep(1)
cal_h, cal_w = load_images_from_folder('cal')
print(cal_h, cal_w)

time.sleep(1)
print("\n\n ---------- Elefant ----------")
time.sleep(1)
elefant_h, elefant_w = load_images_from_folder('elefant')
print(elefant_h, elefant_w )

time.sleep(1)
print("\n\n ---------- Fluture ----------")
time.sleep(1)
fluture_h, fluture_w = load_images_from_folder('fluture')
print(fluture_h, fluture_w)

time.sleep(1)
print("\n\n ---------- Gaina ----------")
time.sleep(1)
gaina_h, gaina_w = load_images_from_folder('gaina')
print(gaina_h, gaina_w)

time.sleep(1)
print("\n\n ---------- Oaie ----------")
time.sleep(1)
oaie_h, oaie_w = load_images_from_folder('oaie')
print(oaie_h, oaie_w)

time.sleep(1)
print("\n\n ---------- Paiangen ----------")
time.sleep(1)
paiangen_h, paiangen_w = load_images_from_folder('paiangen')
print(paiangen_h, paiangen_w)

time.sleep(1)
print("\n\n ---------- Pisica ----------")
time.sleep(1)
pisica_h, pisica_w = load_images_from_folder('pisica')
print(pisica_h, pisica_w)

time.sleep(1)
print("\n\n ---------- Vaca ----------")
time.sleep(1)
vaca_h, vaca_w = load_images_from_folder('vaca')
print(vaca_h, vaca_w)

time.sleep(1)
print("\n\n ---------- Veverita ----------")
time.sleep(1)
veverita_h, veverita_w = load_images_from_folder('veverita')
print(veverita_h, veverita_w)

time.sleep(2)
print((caine_h + cal_h + elefant_h + fluture_h + gaina_h + oaie_h + paiangen_h + pisica_h + vaca_h + veverita_h)//10, (caine_w + caine_w + elefant_w + fluture_w + gaina_w + oaie_w + paiangen_w + pisica_w + vaca_w + veverita_w)//10 )
print((caine_h + cal_h + elefant_h + fluture_h + gaina_h + oaie_h + paiangen_h + pisica_h + vaca_h + veverita_h)/10, (caine_w + caine_w + elefant_w + fluture_w + gaina_w + oaie_w + paiangen_w + pisica_w + vaca_w + veverita_w)/10 )