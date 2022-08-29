import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
import time 
plt.show()


# ---------------------------------------------------- #
class Conv_op:
    
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size * filter_size) # impartim pentru a normaliza coeficientii
    
    def image_region(self, image): # extragem bucati din imaginea originala pentru a aplica kernelul
        height, width = image.shape
        self.image = image
        for j in range(height - self.filter_size + 1):
            for k in range(width - self.filter_size + 1):
                image_patch = image[ j : (j + self.filter_size), k:( k + self.filter_size)]
                yield image_patch, j, k
    def forward_prop(self, image):
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for image_patch, i, j in self.image_region(image):
            conv_out[i,j] = np.sum(image_patch * self.conv_filter, axis = (1,2))
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


# ---------------------------------------------------- #

# ---------------------------------------------------- #

class Max_Pool:
    def __init__(self, filter_size):
        self.filter_size = filter_size
    
    def image_region(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width = image.shape[1] // self.filter_size
        self.image = image
        
        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[( i * self.filter_size) : ( i * self.filter_size + self.filter_size), ( j * self.filter_size) : ( j * self.filter_size + self.filter_size)]
                yield image_patch, i, j
    
    def forward_prop(self, image):
        height, width, num_filters = image.shape
        output = np.zeros((height // self.filter_size, width // self.filter_size, num_filters))
        for image_patch, i, j in self.image_region(image):
            output[i, j] = np.amax(image_patch, axis = (0, 1))
            
        return output
    
    def back_prop(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image. shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filters = image_patch.shape
            maximum_val = np.amax(image_patch, axis = (0, 1))
            
            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(num_filters):
                        if image_patch[i1, j1, k1] == maximum_val[k1]:
                            dL_dmax_pool[i * self.filter_size + i1, j * self.filter_size + j1, k1] = dL_dout[i,j,k1]
        
        return dL_dmax_pool

# ---------------------------------------------------- #


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

import mnist

timeStart = time.time()

train_imgs = mnist.train_images()
# train_imgs = train_imgs[:15000,:,:]
train_labels = mnist.train_labels()
# train_labels = train_labels[:15000]
test_imgs = mnist.test_images()
# test_imgs = test_imgs[:15000,:,:]
test_labels  = mnist.test_labels()
# test_labels = test_labels[:15000]

conv = Conv_op(6,6)             #28x28x1 -> 26x26x8
conv_cop = conv
pool = Max_Pool(2)              #26x26x8 -> 13x13x8
pool_cop = pool
softmax = Softmax(121*6,10)   #13x13x8 -> 10
softmax_cop = softmax



def cnn_forward_prop(image, label):
    out_p = conv.forward_prop((image / 255) - 0.5)
    out_p = pool.forward_prop(out_p)
    out_p = softmax.forward_prop(out_p)
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
    grad_back = softmax.back_prop(gradient, learn_rate)

    grad_back = pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learn_rate)
    return loss, acc



#Shuffle the train data
shuffle_data = np.random.permutation(len(train_imgs))
train_imgs = train_imgs[shuffle_data]
train_labels = train_labels[shuffle_data]

#Train the CNN
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(train_imgs, train_labels)):
    if i % 100 == 0:
        print(' %d steps out of 100 steps: Average Loss %.3f and ACCuracy: %d%%' %(i+1,loss/100, num_correct))
        loss = 0
        num_correct = 0
    l1, accu = training_cnn(im, label)
    loss += l1
    num_correct += accu

print("\nTesting Phase\n")


loss = 0 
num_correct = 0
for im, label in zip(test_imgs, test_labels):
    _, l1, acc = cnn_forward_prop(im, label)
    loss += l1
    num_correct += acc

# print(len(test_imgs))
num_tests = len(test_imgs)
print("Test Accuracy: ", num_correct/num_tests)

print("numar imagini antrenare: ", len(train_imgs))
print("numar imagini testare: ", len(test_imgs))

print("Time: ", time.time() - timeStart)