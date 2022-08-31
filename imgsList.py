import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pickle 
from skimage import io

#-----------------Training--------------------

# def load_images_from_folder(folder, list):
#     i = 0
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename),  cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # print(img.shape)
#         w = img.shape[0]
#         h = img.shape[1]
#         c = img.shape[2]
#         img = img.reshape((c,w,h))
#         # print(img.shape)
#         # print("img type:", type(filename))
#         if img is not None:
#             list.append(img)
#             i += 1
#             # plt.imshow(img, cmap = 'gray')
#             # plt.show()
#         # if i == 1800:
#         #     break
#     return list
# list = []
# list = load_images_from_folder('Train/caine',list)
# print("done caine")
# list = load_images_from_folder('Train/cal',list)
# print("done cal")
# list = load_images_from_folder('Train/elefant',list)
# print("done elefant")
# list = load_images_from_folder('Train/fluture',list)
# print("done fluture")
# list = load_images_from_folder('Train/gaina',list)
# print("done gaina")
# list = load_images_from_folder('Train/oaie',list)
# print("done oaie")
# list = load_images_from_folder('Train/paiangen',list)
# print("done paiangen")
# list = load_images_from_folder('Train/pisica',list)
# print("done pisica")
# list = load_images_from_folder('Train/vaca',list)
# print("done vaca")
# list = load_images_from_folder('Train/veverita',list)
# print("done veverita")

# print("lungime lista: ", len(list))
# # for i in range(len(list)):
# #     # list[i] = np.reshape(list[i], list[i].shape[0],list[i].shape[1])
# #     list[i] = list[i][:,:,0]/255+list[i][:,:,1]/255 + list[i][:,:,2]/255

# # for i in range(len(list)):
# #     # list[i] = np.reshape(list[i], list[i].shape[0],list[i].shape[1])
# #     list[i] = list[i][:,:,0]/255+list[i][:,:,1]/255


# list= np.array(list)


# list1_1 = np.zeros(len(list))
# # print(list[1].shape)
# # print(type(list))
# label = -1
# for i in range(len(list1_1)):
#     if i % 1800 == 0:
#         label += 1
#     list1_1[i] = label
# list1_1 = np.array(list1_1)
# list1_1 = list1_1.astype(int)

# print(type(list1_1))

# file = open('img_list', 'wb')
# file1 = open('label_list', 'wb')

# pickle.dump(list, file)
# pickle.dump(list1_1, file1)

# file.close()
# file1.close()

# ---------------Testing------------------
list1 = []
list1_2 = []

# def load_images_from_folder_by_name(folder,c, list,list1):    cls
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename),  cv2.IMREAD_COLOR)
#         # plt.imshow(img, cmap="color")
#         # plt.show()
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # print(img.shape)
#         w = img.shape[0]
#         h = img.shape[1]
#         c = img.shape[2]
#         img = img.reshape((c,w,h))
#         # print(img.shape)
#         if img is not None:
#             list.append(img)
#             list1.append(int(c))

#     return list, list1
dogg = img = io.imread("./flickr_dog_000002.jpg")
plt.imshow(dogg)
plt.show()

print(dogg.shape)

def load_images_from_folder_by_name(folder,c, list,list1):
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder,filename))
        # plt.imshow(img, cmap="color")
        # plt.show()
        # print(img.shape)
        w = img.shape[0]
        h = img.shape[1]
        c = img.shape[2]
        img = img.reshape((c,w,h))
        # print(img.shape)
        if img is not None:
            list.append(img)
            list1.append(int(c))

    return list, list1

list1, list1_2 = load_images_from_folder_by_name('Test/caine','0',list1, list1_2)
print("done caine")
list1, list1_2 = load_images_from_folder_by_name('Test/cal','1',list1, list1_2)
print("done cal")
list1, list1_2 = load_images_from_folder_by_name('Test/elefant','2',list1, list1_2)
print("done elefant")
list1, list1_2= load_images_from_folder_by_name('Test/fluture','3',list1, list1_2)
print("done fluture")
list1, list1_2 = load_images_from_folder_by_name('Test/gaina','4',list1, list1_2)
print("done gaina")
list1, list1_2 = load_images_from_folder_by_name('Test/oaie','5',list1, list1_2)
print("done oaie")
list1, list1_2 = load_images_from_folder_by_name('Test/paiangen','6',list1, list1_2)
print("done paiangen")
list1, list1_2 = load_images_from_folder_by_name('Test/pisica','7',list1, list1_2)
print("done pisica")
list1, list1_2 = load_images_from_folder_by_name('Test/vaca','8',list1, list1_2)
print("done vaca")
list1, list1_2 = load_images_from_folder_by_name('Test/veverita','9',list1, list1_2)
print("done veverita")

list1 = np.array(list1)

print("l litsa: ", len(list1))
# print(type(list1))
list1_2 = np.array(list1_2)
list1_2 = list1_2.astype(int)

file = open('img_test_list', 'wb')
file1 = open('label_test_list', 'wb')

pickle.dump(list1, file)
pickle.dump(list1_2, file1)

file.close()
file1.close()
