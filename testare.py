from images import testImages, testLabels
from app import cnn_forward_prop, lungime_testare

print("\nTesting Phase\n")


loss = 0 
num_correct = 0
for im, label in zip(testImages, testLabels):
    _, l1, acc = cnn_forward_prop(im, label)
    loss += l1
    num_correct += acc

# print(len(test_imgs))

print("Test Accuracy: ", num_correct/lungime_testare)


print("numar imagini testare: ", lungime_testare)

