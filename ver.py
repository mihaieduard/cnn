from cgi import test
import pickle
import cv2
import numpy as np

# label_list_file = open('label_list', 'rb')

# labels = pickle.load(label_list_file)

# print(type(labels))
# print(labels[1800])

# label_test_list_file = open('label_test_list', 'rb')
# test_labels = pickle.load(label_test_list_file)

# print(type(test_labels))
# print(test_labels[1])

# x = np.random.randn(3 * 3, 3, 3)/(3 * 3)
# print(x.shape[0])
# print(x)
# j = 0
# print('-----------------------------------------------')
# for i in range(x.shape[0]):
#     print(x[i,:,:])
#     j += 1
#     if j == x.shape[0]//3:
#         print("---------")
#         j = 0

# def image_region(image):
#     height, width = image.shape
#     for j in range(height - 3 + 1):
#         for k in range(width - 3 + 1):
#             image_patch = image[ j : (j + 3), k:( k + 3)]
#             yield image_patch, j, k

# conv_filter_Red = np.random.randn(5, 3, 3)/(3 * 3)
# conv_filter_Green = np.random.randn(5, 3, 3)/(3 * 3)
# conv_filter_Blue = np.random.randn(5, 3, 3)/(3 * 3)

# height, width, depth = 10,10,3

# conv_out_red = np.zeros((height - 3 + 1, width - 3 + 1, 5))
# conv_out_green = np.zeros((height - 3 + 1, width - 3 + 1, 5))
# conv_out_blue = np.zeros((height - 3 + 1, width - 3 + 1, 5))
# conv_out = np.zeros((height - 3 + 1, width - 3 + 1, 5))

# image_red = np.random.randn(3,3)
# print(image_red.shape)
# image_green = np.random.randn(3,3)
# image_blue = np.random.randn(3,3)

# for image_patch, i, j in image_region(image_red):
#     conv_out_red[i,j] = np.sum(image_patch * conv_filter_Red, axis = (1,2))
# print(conv_out_red.shape)
# for image_patch, i, j in image_region(image_green):
#     conv_out_green[i,j] = np.sum(image_patch * conv_filter_Green, axis = (1,2))
# print(conv_out_green.shape)
# for image_patch, i, j in image_region(image_blue):
#     conv_out_blue[i,j] = np.sum(image_patch * conv_filter_Blue, axis = (1,2))
# print(conv_out_blue.shape)

# conv_out = np.sum()

x = np.random.randint(0,high = 255, size = (5, 3, 3))/(3*3)
x = np.random.rand(5, 3, 3)/(3*3)
print(x)