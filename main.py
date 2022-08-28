import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
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
        height, width = image.shape[0], image.shape[1]
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

        
        return dL_d_inp.reshape(self.orig_im_shape)


# --------------------- Training --------------- #

def load_images_from_folder(folder, list):
    i = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            img = cv2.resize(img, (h//2, w//2))
            list.append(img)
            i += 1
    return list 


train_imgs = []
train_imgs = load_images_from_folder('Train/caine',train_imgs)
print("done caine")
train_imgs = load_images_from_folder('Train/cal',train_imgs)
print("done cal")
train_imgs = load_images_from_folder('Train/elefant',train_imgs)
print("done elefant")
train_imgs = load_images_from_folder('Train/fluture',train_imgs)
print("done fluture")
train_imgs = load_images_from_folder('Train/gaina',train_imgs)
print("done gaina")
train_imgs = load_images_from_folder('Train/oaie',train_imgs)
print("done oaie")
train_imgs = load_images_from_folder('Train/paiangen',train_imgs)
print("done paiangen")
train_imgs = load_images_from_folder('Train/pisica',train_imgs)
print("done pisica")
train_imgs = load_images_from_folder('Train/vaca',train_imgs)
print("done vaca")
train_imgs = load_images_from_folder('Train/veverita',train_imgs)
print("done veverita")

print("lungime lista: ", len(train_imgs))
train_imgs= np.array(train_imgs)


train_labels = np.zeros(len(train_imgs))
label = -1
for i in range(len(train_labels)):
    if i % 1800 == 0:
        label += 1
    train_labels[i] = label
list
train_labels = np.array(train_labels)
train_labels = train_labels.astype(int)

# ---------------Testing------------------
test_imgs = []
test_labels = []

def load_images_from_folder_by_name(folder,c, list,list12):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            img = cv2.resize(img, (h//2, w//2))
            list12.append(int(c))
            list.append(img)
    return list, list12

test_imgs, test_labels = load_images_from_folder_by_name('Test/caine','0',test_imgs, test_labels)
print("done caine")
test_imgs, test_labels = load_images_from_folder_by_name('Test/cal','1',test_imgs, test_labels)
print("done cal")
test_imgs, test_labels = load_images_from_folder_by_name('Test/elefant','2',test_imgs, test_labels)
print("done elefant")
test_imgs, test_labels= load_images_from_folder_by_name('Test/fluture','3',test_imgs, test_labels)
print("done fluture")
test_imgs, test_labels = load_images_from_folder_by_name('Test/gaina','4',test_imgs, test_labels)
print("done gaina")
test_imgs, test_labels = load_images_from_folder_by_name('Test/oaie','5',test_imgs, test_labels)
print("done oaie")
test_imgs, test_labels = load_images_from_folder_by_name('Test/paiangen','6',test_imgs, test_labels)
print("done paiangen")
test_imgs, test_labels = load_images_from_folder_by_name('Test/pisica','7',test_imgs, test_labels)
print("done pisica")
test_imgs, test_labels = load_images_from_folder_by_name('Test/vaca','8',test_imgs, test_labels)
print("done vaca")
test_imgs, test_labels = load_images_from_folder_by_name('Test/veverita','9',test_imgs, test_labels)
print("done veverita")
print(len(test_imgs))
test_imgs = np.array(test_imgs)
test_labels = np.array(test_labels)
test_labels = test_labels.astype(int)

# --------------------- Creare obiecte convolutie ------------------------------ #

conv = Conv_op(10,9)             #178 x 162 x 1 -> 170 x 154 x 10
conv_cop = conv
pool = Max_Pool(2)              #170 x 154 x 10 -> 85 x 77 x 10
pool_cop = pool
softmax = Softmax(85*77*10,10)   #85 x 77 x 10 -> 10
softmax_cop = softmax

def cnn_forward_prop(image, label):
    out_p = conv.forward_prop((image / 255) - 0.5)
    out_p = pool.forward_prop(out_p)
    out_p = softmax.forward_prop(out_p)
    cross_ent_loss = -np.log(out_p[label])
    accuracy_val = 1 if np.argmax(out_p) == label else 0

    return out_p, cross_ent_loss, accuracy_val

def training_cnn(image, label, learn_rate = 0.005):

    
    out, loss, acc = cnn_forward_prop(image, label)
    
    gradient = np.zeros(10)
    gradient[label] = -1/out[label]


    grad_back = softmax.back_prop(gradient, learn_rate)

    grad_back = pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learn_rate)

    return loss, acc


norms = []
for epoch1 in range(4):
    print('Epoch %d ---->' %(epoch1 + 1))
    

    shuffle_data = np.random.permutation(len(train_imgs))
    train_imgs = train_imgs[shuffle_data]
    train_labels = train_labels[shuffle_data]

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_imgs, train_labels)):
        # print("i: ", i)
        if i % 100 == 0:
            print(' %d steps out of 100 steps: Average Loss %.3f and ACCuracy: %d%%' %(i+1,loss/100, num_correct))
            loss = 0
            num_correct = 0

        l1, accu = training_cnn(im, label)
        loss += l1
        num_correct += accu


# ------------------------------------------------------------------------------------------------------------ #

print("Testing Phase")


loss = 0 
num_correct = 0
for im, label in zip(test_imgs, test_labels):
    _, l1, acc = cnn_forward_prop(im, label)
    loss += l1
    num_correct += acc

print(len(test_imgs))
num_tests = len(test_imgs)
print("Test Accuracy: ", num_correct/num_tests)

'''
Probleme:
1) trebuie rulat de fiecare codul complet, inclusiv partea de training, pentru a ajuge sa testam reteaua
2) putem folosi doar o metoda de optimizare, SGD in cazul de fata, intrucat nu putem salva valorile intiale
3) nu salvam imaginile in fisiere si trebuiesc citite de fiecare data si aduaga timp in plus la rulare
4) un singur pas de convolutie si unu de pooling iar reteaua fully contected nu are hidden layerscls

'''