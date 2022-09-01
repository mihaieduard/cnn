from re import L
import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import pickle
from skimage import io
import time
# from images import trainImages, trainLabels, testImages, testLabels

# dog1 = cv2.imread("flickr_dog_000002.jpg", cv2.IMREAD_COLOR)
# dog1 = cv2.cvtColor(dog1, cv2.COLOR_BGR2RGB)
# plt.imshow(dog1)
# plt.show()

# print(dog1.shape)
# print("Red")
# print(dog1[:,:,0])
# print("Green")
# print(dog1[:,:,1])
# print("Blue")
# print(dog1[:,:,2])

# print("\n\n")
# print("image")
# print(dog1)


# w = dog1.shape[0]
# h = dog1.shape[1]
# c = dog1.shape[2]
# dog1 = dog1.reshape((c,w,h))

# print(dog1.shape)
# print("Red")
# print(dog1[0,:,:])
# print("Green")
# print(dog1[1,:,:])
# print("Blue")
# print(dog1[2,:,:])

# print("\n\n")
# print("image")
# print(dog1)
# de facut un reshape la 3 x 357 x 325


# class Convolution:
    
#     def __init__(self, numberOfFilters, filtersSize):
#         self.numberOfFilters = numberOfFilters
#         self.filtersSize = filtersSize
#         self.convolutionFilterRed = np.random.randn(numberOfFilters, filtersSize, filtersSize)/(filtersSize*filtersSize) # impartim pt normalizarea coef
#         self.convolutionFilterGreen = np.random.randn(numberOfFilters, filtersSize, filtersSize)/(filtersSize*filtersSize) # impartim pt normalizarea coef
#         self.convolutionFilterBlue = np.random.randn(numberOfFilters, filtersSize, filtersSize)/(filtersSize*filtersSize) # impartim pt normalizarea coef

#         self.convolutionFilterRed = round(self.convolutionFilterRed, 3)
#         self.convolutionFilterGreen = round(self.convolutionFilterGreen, 3)
#         self.convolutionFilterBlue = round(self.convolutionFilterBlue, 3)

#     def forwardProp(self, image):
#         height, width,channels = image.shape
#         CR = np.zeros((self.numberOfFilters, height - self.filtersSize + 1, width - self.filtersSize + 1))
#         CG = np.zeros((self.numberOfFilters, height - self.filtersSize + 1, width - self.filtersSize + 1))
#         CB = np.zeros((self.numberOfFilters, height - self.filtersSize + 1, width - self.filtersSize + 1))

class Convolution:
    
    def __init__(self, numberOfFilters, filtersSize):
        self.numberOfFilters = numberOfFilters
        self.filtersSize = filtersSize
        # c = image.shape[numberOfFilters0]
        # self.filtre = {}
        # for i in range(c):
        #     self.filtre["ConvFilter{}".format(i)] = np.random.randn(numberOfFilters, filtersSize, filtersSize)/(filtersSize*filtersSize)
        # for i in range(c):
        #     for x in range(self.numberOfFilters):
        #         for j in range(self.filtersSize):
        #             for k in range(self.filtersSize):
        #                 self.filtre["ConvFilter{}".format(i)][x][j][k] = round(self.filtre["ConvFilter{}".format(i)][x][j][k],3)
        # self.filtre = np.random.randn(numberOfFilters*c, filtersSize, filtersSize)/(filtersSize*filtersSize)
        # for i in range(numberOfFilters*c):
        #     for j in range(filtersSize):
        #         for k in range(filtersSize):
        #             self.filtre[i][j][k] = round(self.filtre[i][j][k], 3)

        # self.filtre = []
        # for i in range(c*numberOfFilters):
        #     self.filtre.append(np.random.randn(filtersSize, filtersSize)/(filtersSize*filtersSize))

    def forwardProp(self,image):
        # self.image = image
        # channels, height, width = self.image.shape
        # self.Conv = {}
        # self.RezConv = {}
        # k=0
        # for i in range(channels * self.numberOfFilters):
        #     self.Conv["C{}".format(i)] = np.zeros((self.numberOfFilters, height - self.filtersSize + 1, width - self.filtersSize + 1))
        # for i in range(self.numberOfFilters):
        #     self.RezConv["Rez{}".format(i)] = np.zeros((self.numberOfFilters, height - self.filtersSize + 1, width - self.filtersSize + 1))
        # for i in range(channels):
        #     # for j in range(i*self.numberOfFilters, i*self.numberOfFilters + self.numberOfFilters):
        #     for j in range(self.numberOfFilters):
        #         self.Conv['C{}'.format(k)] = signal.correlate2d(self.image[i,:,:], self.filtre["ConvFilter{}".format(i)][j], "valid")
        #         self.RezConv["Rez{}".format(j)] += self.Conv['C{}'.format(k)]
        #         k += 1
        
        # return self.Conv, self.RezConv
        c = image.shape[0]
        self.filtre = np.random.randn(self.numberOfFilters*c, self.filtersSize, self.filtersSize)/(self.filtersSize*self.filtersSize)
        
        for i in range(self.numberOfFilters*c):
            for j in range(self.filtersSize):
                for k in range(self.filtersSize):
                    self.filtre[i][j][k] = round(self.filtre[i][j][k], 3)
        channels, height, width = image.shape
        self.Conv = np.zeros((channels * self.numberOfFilters, height - self.filtersSize + 1, width - self.filtersSize + 1))
        self.RezConv = np.zeros((self.numberOfFilters, height - self.filtersSize + 1, width - self.filtersSize + 1))
        k = 0
        for i in range(channels):
            for j in range(self.numberOfFilters):
                self.Conv[k] = signal.correlate2d(image[i,:,:], self.filtre[i+j], "valid")
                self.RezConv[j] += self.Conv[k]
                k+=1
        self.image = image
        return self.RezConv
    def backProp(self, dL_dmax_pool, learning_rate):
        # print("self.image.shape: ", self.image.shape)
        # print("dL_dmax_pool.shape: ", dL_dmax_pool.shape)
        # print("self.image.shape[0]: ", self.image.shape[0],"\nself.image.shape[1]: ", self.image.shape[1],"\nself.image.shape[2] :", self.image.shape[2])
        dL_dX = np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        dL_dK = np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        # print("dL_dX.shape: ",dL_dX.shape)
        # print("dL_dmax_pool.shape[0]: ", dL_dmax_pool.shape[0])
        # print("self.filtre.shape: ", self.filtre.shape)
        # print("numberOfFilters: ", self.numberOfFilters)
        # print("dL_dmax_pool[0,:,:].shape: ",dL_dmax_pool[0,:,:].shape)
        # print("self.filtre[0][0].shape: ", self.filtre[0+0].shape)
        # print("self.image[i,:,:].sahpe: ",self.image.shape)
        # print("dL_dmax_pool[i,:,:].shape: ", dL_dmax_pool.shape)
        for i in range(self.image.shape[0]):
            for j in range(self.numberOfFilters):
                dL_dX[i] = signal.convolve2d(dL_dmax_pool[i,:,:], self.filtre[i+j], "full")
        # print("dL_dX.shape", dL_dX.shape)
        # print("dL_dK.shape", dL_dK.shape)

        # dL_dF_params = np.zeros(self.filtre.shape)
        # for i in range(self.filtre.shape[0]):
        #     dL_dF_params[i] = signal.correlate2d()

        dL_dK = np.zeros(self.filtre.shape)
        # print(dL_dK.shape)
        # print(self.filtre.shape)
        # print(self.numberOfFilters)
        # print(dL_dK[10])
        # print("self.image.shape: ", self.image.shape)
        # print("dL_dmax_pool: ", dL_dmax_pool.shape)
        k=0
        for i in range(self.image.shape[0]):
            for j in range(self.numberOfFilters):
                # print("i: ",i)
                # print("j: ",j)
                # print("k: ",k)
                # dL_dK[k] =  np.copy(signal.correlate2d(self.image[j,:,:],dL_dmax_pool[i*self.image.shape[0]+j], "valid"))
                dL_dK[k] =  np.copy(signal.correlate2d(self.image[i,:,:],dL_dmax_pool[j], "valid"))
                # print("dL_dK[k.shape]: ", dL_dK[k].shape)
                k += 1
        #         print("\n-------------\n")
        # print("dL_dK.shape: ",dL_dK.shape)

        self.filtre -= learning_rate * dL_dK

        return dL_dX
    

class Pooling():

    def __init__(self, filter_size):
        self.filter_size = filter_size

    def forward_prop(self, image):
        self.image = image
        self.num_filters, self.height, self.width  = image.shape
        self.output = np.zeros((self.num_filters, self.height // self.filter_size, self.width // self.filter_size))
        for k in range(self.num_filters):
            for i in range(self.height//self.filter_size):
                for j in range(self.width//self.filter_size):
                    sub = image[k, i*self.filter_size: i*self.filter_size + self.filter_size, j*self.filter_size: j*self.filter_size + self.filter_size]
                    self.output[k,i,j] = np.amax(sub, axis = (0, 1))

        return self.output

    def back_prop_final(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        # print("self.num_filters: ",self.num_filters)
        # print("dL_dout.shape: ", dL_dout.shape)
        for k in range(self.num_filters):
            for i in range(self.height//self.filter_size):
                for j in range(self.width//self.filter_size):
                    sub = self.image[k, i*self.filter_size: i*self.filter_size + self.filter_size, j*self.filter_size: j*self.filter_size + self.filter_size]
                    height, width  = sub.shape
                    maximum_val = np.amax(sub, axis = (0, 1))
                    # for k1 in range(num_filters):
                    for i1 in range(height):
                        for j1 in range(width):
                            if sub[i1, j1] == maximum_val:
                                dL_dmax_pool[k, i * self.filter_size + i1, j * self.filter_size + j1] = dL_dout[k, i,j]
        
        return dL_dmax_pool #dL_dZ
        
## de citiit lungimea dict si parcurs fiecrae element pt a doua convolutie


class Softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node) / input_node
        self.bias = np.zeros(softmax_node)
    
    def forward_prop(self, image):
        self.orig_im_shape = image.shape
        image_modified = image.flatten()
        self.modified_input = image_modified
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        return exp_out/np.sum(exp_out, axis = 0)
    
    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue
            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)

            #Gradients with respect to out(z)
            dy_dz = -transformation_eq[i] * transformation_eq / (S_total ** 2)
            dy_dz[i] = transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total ** 2)

            #Gradiens of totals against weights/biases/inputs
            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight

            #Gradients of loss against totals
            dL_dz = grad * dy_dz

            #Gradients of loss against weights/biases/input
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
#             print(dL_dw.shape)
            dL_db = dL_dz * dz_db
#             print(dL_dw.shape)
            dL_d_inp =  dz_d_inp @ dL_dz
#             print(dL_d_inp.shape)
        
        #Update weights and biases
        self.weight -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db
        # pickle.dump( self.bias, open( "softmax_bias.p", "wb" ) )
        # pickle.dump( self.weight, open( "softmax_weight.p", "wb" ) )
        
        return dL_d_inp.reshape(self.orig_im_shape)
        
# img = io.imread('flickr_dog_000002.jpg')
# h,w,c = img.shape

# img = img.reshape((c,h,w))

# con = Convolution(5, 3, img)
# Conv, con_RezConv = con.forwardProp()

# con1 = Convolution(10,3, con_RezConv)
# _,con1_RezConv = con1.forwardProp()
# print(Conv)
# print("\n\n---------------------------------------------------\n\n")
# print(con1_RezConv.shape)

# print("\n\n---------------------------------------------------\n\n")
# pool = Pooling(2, con1_RezConv)
# pool_fp = pool.forward_prop()
# print(pool_fp.shape)

# print(con.filtre["ConvFilter{}".format(0)])
# print("\n\n---------------------------------------------------\n\n")
# print(round(con.filtre["ConvFilter{}".format(0)][0][0][0],3))
# print("\n\n---------------------------------------------------\n\n")
# print(con.filtre)
# print("\n\n---------------------------------------------------\n\n")
# Conv, RezConv = con.forwardProp(img)
# print(Conv)
# print("\n\n---------------------------------------------------\n\n")
# print(RezConv)

# conv = Convolution(6,6,img)             #28x28x1 -> 23x23x6
# # conv_cop = conv
# pool = Pooling(2,conv.forwardProp())              #23x23x6 -> 11x11x6
# # pool_cop = pool
# conv2 = Convolution(10,2, pool.forward_prop())
# pool2 = Pooling(2, conv2.forwardProp())
# print(pool2.forward_prop().shape)
# softmax = Softmax(pool2.forward_prop().shape[0] *pool2.forward_prop().shape[1]*pool2.forward_prop().shape[2],10)   #13x13x8 -> 10
# print(softmax.forward_prop(pool2.forward_prop()))


conv1 = Convolution(6,6) # 3 x 325 x 357 ->6 x 320 x 352 
pool1 = Pooling(2) # -> 6 x 160 x 176
conv2 = Convolution(10,8)   # -> 10 x 153 x 169 
pool2 = Pooling(2)  # -> 10 X 76 X 84
soft  = Softmax(10 * 76 * 84,10) # -> 10
# print(img.shape)






def cnn_forward_prop(image, label):

    out_p = conv1.forwardProp((image / 255) - 0.5)
    out_p = pool1.forward_prop(out_p)
    # print(out_p.shape)
    out_p = conv2.forwardProp(out_p)
    out_p = pool2.forward_prop(out_p)
    out_p = soft.forward_prop(out_p)
#     print(out_p)
    #Calculate cross-entropy loss and accuracy
    cross_ent_loss = -np.log(out_p[label])
    accuracy_val = 1 if np.argmax(out_p) == label else 0
#     print(out_p.shape)
    return out_p, cross_ent_loss, accuracy_val

def training_cnn(image, label, learn_rate = 0.005):
    #Forward
    
    out, loss, acc = cnn_forward_prop(image, label)
    
    #Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1/out[label]
#     print(label, out[label], gradient[label])
#     print(gradient)
    
    #Backprop
    grad_back = soft.back_prop(gradient, learn_rate)

    grad_back = pool2.back_prop_final(grad_back)
    grad_back = conv2.backProp(grad_back, 0.5)
    # print("grad_back.shape: ",grad_back.shape)
    grad_back = pool1.back_prop_final(grad_back)
    grad_back = conv1.backProp(grad_back, 0.5)
    # print("grad_back.shape: ",grad_back.shape)
    return loss, acc

