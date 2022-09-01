from re import L
import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import pickle
from skimage import io
import time
from images import trainImages, trainLabels, testImages, testLabels

class Convolution:
    
    def __init__(self, numberOfFilters, filtersSize):
        self.numberOfFilters = numberOfFilters
        self.filtersSize = filtersSize

    def forwardProp(self,image, filtre_primite = 0):
        if filtre_primite == 0:
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
        else:
            self.filtre = filtre_primite
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
       
        dL_dX = np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        dL_dK = np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        for i in range(self.image.shape[0]):
            for j in range(self.numberOfFilters):
                dL_dX[i] = signal.convolve2d(dL_dmax_pool[i,:,:], self.filtre[i+j], "full")
        dL_dK = np.zeros(self.filtre.shape)
        k=0
        for i in range(self.image.shape[0]):
            for j in range(self.numberOfFilters):
                
                dL_dK[k] =  np.copy(signal.correlate2d(self.image[i,:,:],dL_dmax_pool[j], "valid"))
                k += 1
        self.filtre -= learning_rate * dL_dK

        return dL_dX, self.filtre
    

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
        for k in range(self.num_filters):
            for i in range(self.height//self.filter_size):
                for j in range(self.width//self.filter_size):
                    sub = self.image[k, i*self.filter_size: i*self.filter_size + self.filter_size, j*self.filter_size: j*self.filter_size + self.filter_size]
                    height, width  = sub.shape
                    maximum_val = np.amax(sub, axis = (0, 1))
                    for i1 in range(height):
                        for j1 in range(width):
                            if sub[i1, j1] == maximum_val:
                                dL_dmax_pool[k, i * self.filter_size + i1, j * self.filter_size + j1] = dL_dout[k, i,j]
        
        return dL_dmax_pool #dL_dZ


class Softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node) / input_node
        self.bias = np.zeros(softmax_node)
    
    def forward_prop(self, image, weight_primti = 0, bias_primit = 0):
        if weight_primti == 0 and bias_primit == 0:
            self.orig_im_shape = image.shape
            image_modified = image.flatten()
            self.modified_input = image_modified
            output_val = np.dot(image_modified, self.weight) + self.bias
            self.out = output_val
            exp_out = np.exp(output_val)
        else:
            self.orig_im_shape = image.shape
            image_modified = image.flatten()
            self.modified_input = image_modified
            output_val = np.dot(image_modified, weight_primti) + bias_primit
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
        
        return dL_d_inp.reshape(self.orig_im_shape), self.weight, self.bias

timeStart = time.time()



print("see creeaza obiectele")
conv1 = Convolution(12,6) # 3 x 325 x 357 ->6 x 320 x 352 
pool1 = Pooling(2) # -> 6 x 160 x 176
conv2 = Convolution(50,8)   # -> 10 x 153 x 169 
pool2 = Pooling(2)  # -> 10 X 76 X 84
soft  = Softmax(50 * 76 * 84,10) # -> 10

print("se amesteca datele")
shuffle_data = np.random.permutation(len(trainImages))
trainImages = trainImages[shuffle_data]
trainLabels = trainLabels[shuffle_data]

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
    grad_back,w,b = soft.back_prop(gradient, learn_rate)

    grad_back = pool2.back_prop_final(grad_back)
    grad_back, f2 = conv2.backProp(grad_back, 0.5)
    # print("grad_back.shape: ",grad_back.shape)
    grad_back = pool1.back_prop_final(grad_back)
    grad_back, f1 = conv1.backProp(grad_back, 0.5)
    # print("grad_back.shape: ",grad_back.shape)
    return loss, acc, f1, f2, w,b

print("\n ------------------------ Training Phase\n")
loss = 0
lungime_antrenare = len(trainImages)
lungime_testare = len(testImages)
num_correct = 0
for i, (im, label) in enumerate(zip(trainImages, trainLabels)):
    print(i)
    if i % 100 == 0:
        print(' %d steps out of 100 steps: Average Loss %.3f and ACCuracy: %d%%' %(i+1,loss/100, num_correct))
        loss = 0
        num_correct = 0
    l1, accu,f1,f2,w,b = training_cnn(im, label)
    loss += l1
    num_correct += accu
    if i == lungime_antrenare-1:
        print("i in if: ", i)
        print("lungime antrenare - 1: ", lungime_antrenare-1)
        file = open('filtre1', 'wb')
        file1 = open('filtre2', 'wb')
        file2 = open('weight', 'wb')       
        file3 = open('bias', 'wb')   

        pickle.dump(file, f1)
        pickle.dump(file1, f2)
        pickle.dump(file2, w)
        pickle.dump(file3, b)

        file.close()   
        file1.close()   
        file2.close()   
        file3.close()   

print("numar imagini antrenare: ", lungime_antrenare)

# print("\nTesting Phase\n")


# loss = 0 
# num_correct = 0
# for im, label in zip(testImages, testLabels):
#     _, l1, acc = cnn_forward_prop(im, label)
#     loss += l1
#     num_correct += acc

# # print(len(test_imgs))

# print("Test Accuracy: ", num_correct/lungime_testare)

# print("numar imagini antrenare: ", lungime_antrenare)
# print("numar imagini testare: ", lungime_testare)

print("Time: ", time.time() - timeStart)