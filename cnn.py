from tkinter import image_types
import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
import pickle
import time 
import multiprocessing
from scipy import signal
# img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)/255
# plt.imshow(img,cmap = 'gray')
# plt.show()
# print(img.shape)

img1 = cv2.imread('lena.png', cv2.IMREAD_COLOR)
img1  = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
print(img1.shape)
plt.imshow(img1)
plt.show()


class Convolutie:

    def __init__(self, NumberOfFilers, SizeOfFilter):
        self.NumberOfFilers = NumberOfFilers
        self.SizeOfFilter = SizeOfFilter
        self.conv_filter_Red = np.random.rand(NumberOfFilers, SizeOfFilter, SizeOfFilter)/(SizeOfFilter * SizeOfFilter)
        self.conv_filter_Green = np.random.rand(NumberOfFilers, SizeOfFilter, SizeOfFilter)/(SizeOfFilter * SizeOfFilter)
        self.conv_filter_Blue = np.random.rand(NumberOfFilers, SizeOfFilter, SizeOfFilter)/(SizeOfFilter * SizeOfFilter)
        # self.conv_filter_Red = np.random.randint(0,high = 255, size = (NumberOfFilers, SizeOfFilter, SizeOfFilter))/(SizeOfFilter * SizeOfFilter)
        # self.conv_filter_Green = np.random.randint(0, high = 255, size = (NumberOfFilers, SizeOfFilter, SizeOfFilter))/(SizeOfFilter * SizeOfFilter)
        # self.conv_filter_Blue = np.random.randint(0, high = 255, size = (NumberOfFilers, SizeOfFilter, SizeOfFilter))/(SizeOfFilter * SizeOfFilter)

        print("conv_filter_Red: ", self.conv_filter_Red.shape)
        print("conv_filter_Green: ", self.conv_filter_Green.shape)
        print("conv_filter_Blue: ", self.conv_filter_Blue.shape)

    def image_region(self, image):
        height, width = image.shape
        self.image = image
        for j in range(height - self.SizeOfFilter + 1):
            for k in range(width - self.SizeOfFilter + 1):
                image_patch = image[ j : (j + self.SizeOfFilter), k:( k + self.SizeOfFilter)]
                yield image_patch, j, k

    def forward_prop(self, image):
        height, width, depth = image.shape
        conv_out_red = np.zeros((height - self.SizeOfFilter + 1, width - self.SizeOfFilter + 1, self.NumberOfFilers))
        conv_out_green = np.zeros((height - self.SizeOfFilter + 1, width - self.SizeOfFilter + 1, self.NumberOfFilers))
        conv_out_blue = np.zeros((height - self.SizeOfFilter + 1, width - self.SizeOfFilter + 1, self.NumberOfFilers))
        conv_out = np.zeros((height - self.SizeOfFilter + 1, width - self.SizeOfFilter + 1, self.NumberOfFilers))
        image_red = image[:,:,0]/255
        print("image red shape: ", image_red.shape)
        image_green = image[:,:,1]/255
        print("image green shape: ", image_green.shape)
        image_blue = image[:,:,2]/255
        print("image blue shape: ", image_blue.shape)
        for image_patch, i, j in self.image_region(image_red):
            conv_out_red[i,j] = np.sum(image_patch * self.conv_filter_Red, axis = (1,2))
        print("conv_out_red\n")
        print(conv_out_red)
        for image_patch, i, j in self.image_region(image_green):
            conv_out_green[i,j] = np.sum(image_patch * self.conv_filter_Green, axis = (1,2))
        print("conv_out_green\n")
        print(conv_out_green)
        for image_patch, i, j in self.image_region(image_blue):
            if i == 0 and j == 0:
                print(image_patch.shape)
            conv_out_blue[i,j] = np.sum(image_patch * self.conv_filter_Blue, axis = (1,2))
        print("conv_out_blue\n")
        print(conv_out_blue)

        for i in range(self.NumberOfFilers):
            for j in range(self.SizeOfFilter):
                for k in range(self.SizeOfFilter):
                    conv_out[i,j,k] = conv_out_red[i,j,k] + conv_out_green[i,j,k] + conv_out_blue[i,j,k]
        print("conv_out\n")
        print(conv_out)
        return conv_out


        return conv_out

    def back_prop(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filter.shape)
        for image_patch, i, j in self.image_region(self.image):
            for k in range(self.num_filters):
                dL_dF_params[k] += image_patch * dL_dout[i,j,k]
        
        #Filter params update
        self.conv_filter -= learning_rate * dL_dF_params
        # pickle.dump( self.conv_filter, open( "conv_filter.p", "wb" ) )
        return dL_dF_params

conn = Convolutie(18, 7)
out = conn.forward_prop(img1)
print(out.shape)
for i in range(18):
    plt.imshow(out[:,:,i])
    plt.show()