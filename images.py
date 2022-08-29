import matplotlib.pyplot as plt
import cv2
import os
import csv
import numpy as np

#-----------------Training--------------------

def load_images_from_folder(folder, list):
    i = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        # print("img type:", type(filename))
        if img is not None:
            list.append(img)
            i += 1
        if i == 4600:
            break
    return list
list = []
list = load_images_from_folder('raw_image/training/Cat',list)
print("done")
list = load_images_from_folder('raw_image/training/Cheetah',list)
print("done")
list = load_images_from_folder('raw_image/training/Chimpanzee',list)
print("done")
list = load_images_from_folder('raw_image/training/Coyote',list)
print("done")
list = load_images_from_folder('raw_image/training/Guinea',list)
print("done")
list = load_images_from_folder('raw_image/training/Hamster',list)
print("done")
list = load_images_from_folder('raw_image/training/Jaguar',list)
print("done")
list = load_images_from_folder('raw_image/training/Lynx',list)
print("done")
list = load_images_from_folder('raw_image/training/Orangutan',list)
print("done")
list = load_images_from_folder('raw_image/training/Wolf',list)
print("done")

# print(len(list))
for i in range(len(list)):
    # list[i] = np.reshape(list[i], list[i].shape[0],list[i].shape[1])
    list[i] = list[i][:,:,0]/255+list[i][:,:,1]/255 + list[i][:,:,2]/255

list= np.array(list)

list1_1 = np.zeros(len(list))
# print(list[1].shape)
# print(type(list))
label = -1
for i in range(len(list1_1)):
    if i % 4600 == 0:
        label += 1
    list1_1[i] = label
list1_1 = np.array(list1_1)
# print(type(list1_1))


# ---------------Testing------------------
list1 = []
list1_2 = []

def load_images_from_folder_by_name(folder,c, list,list1):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None and filename.startswith(c):
            list.append(img)
            list1.append(int(c))

    return list, list1

list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','0',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','1',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','2',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','3',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','4',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','5',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','6',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','7',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','8',list1,list1_2)
print("done")
list1,list1_2 = load_images_from_folder_by_name('raw_image/testing','9',list1,list1_2)
print("done")

# print(len(list1))
for i in range(len(list1)):
    # list[i] = np.reshape(list[i], list[i].shape[0],list[i].shape[1])
    list1[i] = list1[i][:,:,0]/255+list1[i][:,:,1]/255 + list1[i][:,:,2]/255\

# print(list1[1].shape)
list1 = np.array(list1)
# print(type(list1))
list1_2 = np.array(list1_2)
# print(type(list1_2))
# print(list1_2.shape)
# print(len(list1_2))
print(type(list1_1[1]))

# x1 = load_images_from_folder('raw_image/training/Cat')
# print("done")
# x2 = load_images_from_folder('raw_image/training/Cheetah')
# print("done")
# x3 = load_images_from_folder('raw_image/training/Chimpanzee')
# print("done")
# x4 = load_images_from_folder('raw_image/training/Coyote')
# print("done")
# x5 = load_images_from_folder('raw_image/training/Guinea')
# print("done")
# x6 = load_images_from_folder('raw_image/training/Hamster')
# print("done")
# x7 = load_images_from_folder('raw_image/training/Jaguar')
# print("done")
# x8 = load_images_from_folder('raw_image/training/Lynx')
# print("done")
# x9 = load_images_from_folder('raw_image/training/Orangutan')
# print("done")
# x10 = load_images_from_folder('raw_image/training/Wolf')
# print("done")

# lst = []
# lst.append(x1)
# lst.append(x2)
# lst.append(x3)
# lst.append(x4)
# lst.append(x5)
# lst.append(x6)
# lst.append(x7)
# lst.append(x8)
# lst.append(x9)
# lst.append(x10)

# lst1 = np.arange(10)
# print(lst1[np.newaxis])

# def load_images_from_folder_by_name(folder,c):
#     images = []
#     i = 0
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None and filename.startswith(c):
#             images.append(img)
#             i += 1
#     return images
# x11= load_images_from_folder_by_name('raw_image/testing','0')
# print("done")
# x22 = load_images_from_folder_by_name('raw_image/testing','1')
# print("done")
# x33 = load_images_from_folder_by_name('raw_image/testing','2')
# print("done")
# x44 = load_images_from_folder_by_name('raw_image/testing','3')
# print("done")
# x55 = load_images_from_folder_by_name('raw_image/testing','4')
# print("done")
# x66 = load_images_from_folder_by_name('raw_image/testing','5')
# print("done")
# x77 = load_images_from_folder_by_name('raw_image/testing','6')
# print("done")
# x88 = load_images_from_folder_by_name('raw_image/testing','7')
# print("done")
# x99 = load_images_from_folder_by_name('raw_image/testing','8')
# print("done")
# x100 = load_images_from_folder_by_name('raw_image/testing','9')
# print("done")

# lst11 = []
# lst11.append(x11)
# lst11.append(x22)
# lst11.append(x33)
# lst11.append(x44)
# lst11.append(x55)
# lst11.append(x66)
# lst11.append(x77)
# lst11.append(x88)
# lst11.append(x99)
# lst11.append(x100)

# print(len(x11)+len(x22)+len(x33)+len(x44)+len(x55)+len(x66)+len(x77)+len(x88)+len(x99)+len(x100))
# with open("out.csv", "w") as f:
#     wr = csv.writer(f)
#     wr.writerows(lst)
# from numpy import block

# file = open("out.csv")
# csvreader = csv.reader(file)



# # print(len(lst))
# plt.imshow(lst[0][1])
# plt.show()

# plt.imshow(csvreader[1])   
# plt.show()

