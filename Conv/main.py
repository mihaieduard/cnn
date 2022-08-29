import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import time

img1 = cv2.imread('lena.png', cv2.IMREAD_COLOR)
img1  = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
print(img1.shape)

def KernelsInit(NumberOfFilters, SizeOfFilter):
    conv_filter_Red = np.random.rand(NumberOfFilters, SizeOfFilter, SizeOfFilter)/(SizeOfFilter * SizeOfFilter)
    conv_filter_Green = np.random.rand(NumberOfFilters, SizeOfFilter, SizeOfFilter)/(SizeOfFilter * SizeOfFilter)
    conv_filter_Blue = np.random.rand(NumberOfFilters, SizeOfFilter, SizeOfFilter)/(SizeOfFilter * SizeOfFilter)

    print("conv_filter_Red: ", conv_filter_Red.shape,"\n", conv_filter_Red)
    print("conv_filter_Green: ", conv_filter_Green.shape)
    print("conv_filter_Blue: ", conv_filter_Blue.shape)

    return conv_filter_Red, conv_filter_Green, conv_filter_Blue
def ImageRegion(image, SizeOfFilter):
    height, width = image.shape
    for j in range(height - SizeOfFilter + 1):
        for k in range(width - SizeOfFilter + 1):
            image_patch = image[ j : (j + SizeOfFilter), k:( k + SizeOfFilter)]
            yield image_patch, j, k

def ForwardPropInit(image,NumberOfFilters, SizeOfFilter):
    height, width, depth= image.shape
    print(height, width, depth)
    conv_out_red = np.zeros((height - SizeOfFilter + 1, width - SizeOfFilter + 1, NumberOfFilters))
    conv_out_green = np.zeros((height - SizeOfFilter + 1, width - SizeOfFilter + 1, NumberOfFilters))
    conv_out_blue = np.zeros((height - SizeOfFilter + 1, width - SizeOfFilter + 1, NumberOfFilters))
    conv_out = np.zeros((height - SizeOfFilter + 1, width - SizeOfFilter + 1, NumberOfFilters))
    conv_filter_Red, conv_filter_Green, conv_filter_Blue = KernelsInit(NumberOfFilters, SizeOfFilter)
    for image_patch, i ,j in ImageRegion(image[:,:,0], SizeOfFilter):
        conv_out_red[i,j] = np.sum(image_patch * conv_filter_Red, axis = (1,2))
    for image_patch, i ,j in ImageRegion(image[:,:,1], SizeOfFilter):
        conv_out_green[i,j] = np.sum(image_patch * conv_filter_Green, axis = (1,2))
    for image_patch, i ,j in ImageRegion(image[:,:,2], SizeOfFilter):
        conv_out_blue[i,j] = np.sum(image_patch * conv_filter_Blue, axis = (1,2))
    conv_out = conv_out_red + conv_out_green + conv_out_blue

    return conv_out

conv_out = ForwardPropInit(img1, 5, 3)
print("conv_out: \n", conv_out)
print("\n:::::: ", conv_out[0][0][0])
plt.imshow(conv_out[:,:,0],cmap = 'gray')
plt.show()
# plt.imshow(conv_out[:,:,1],cmap = 'gray')
# plt.show()
# plt.imshow(conv_out[:,:,2],cmap = 'gray')
# plt.show()
# plt.imshow(conv_out[:,:,3],cmap = 'gray')
# plt.show()
# plt.imshow(conv_out[:,:,4],cmap = 'gray')
# plt.show()

def ImageRegionMaxPool(image, FilterSize):
    new_height = image.shape[0] // FilterSize
    new_width = image.shape[1] // FilterSize

    for i in range(new_height):
        for j in range(new_width):
            image_patch = image[( i * FilterSize) : ( i * FilterSize + FilterSize), ( j * FilterSize) : ( j * FilterSize + FilterSize)]
            yield image_patch, i, j

def ForwardPropMaxPool(image, FilterSize):
    height, width, num_filters = image.shape
    print(height, width, num_filters)
    output = np.zeros((height // FilterSize, width // FilterSize, num_filters))
    for image_patch, i, j in ImageRegionMaxPool(image, FilterSize):
        output[i, j] = np.amax(image_patch, axis = (0, 1))
    
    return output

output = ForwardPropMaxPool(conv_out, 2)
print(output.shape)

plt.imshow(output[:,:,0], cmap = 'gray')
plt.show()

def SoftmaxInit(InputNode, SoftmaxNode):
    weight = np.random.randn(InputNode, SoftmaxNode) / InputNode
    bias = np.zeros(SoftmaxNode)

    return weight, bias

def ForwardPropSoftmax(image, InputNode, SoftmaxNode ):
    orig_im_shape = image.shape
    image_modified = image.flatten()
    modified_input = image_modified
    weight, bias = SoftmaxInit(InputNode, SoftmaxNode)
    output_val = np.dot(image_modified, weight) + bias
    out = output_val
    exp_out = np.exp(output_val)
    return exp_out/np.sum(exp_out, axis = 0)

# conn3 = SoftmaxInit(output, output.shape[0] * output.shape[1] * output.shape[2], 10)