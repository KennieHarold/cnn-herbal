import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np

IMG_SIZE = 150
LR = 1e-3
MODEL = 'models/herbal-0.001-6conv-basic.model'
TEST_DATA = 'out/test_data.npy'

def load_model():
    convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input')
    
    convnet = conv_2d(convnet, 32, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation ='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 5, activation ='softmax')
    convnet = regression(convnet, optimizer ='adam', learning_rate = LR, loss ='categorical_crossentropy', name ='targets')

    model = tflearn.DNN(convnet, tensorboard_dir ='log')
    model.load(MODEL)
    
    return model
    
test_data = np.load(TEST_DATA, allow_pickle=True)
model = load_model()
fig = plt.figure()

for num, data in enumerate(test_data[:20]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4, 5, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0: 
        str_label ='lagundi'
    elif np.argmax(model_out) == 1: 
        str_label ='mayana'
    elif np.argmax(model_out) == 2: 
        str_label ='oregano'
    elif np.argmax(model_out) == 3: 
        str_label ='sambong'
    elif np.argmax(model_out) == 4: 
        str_label ='yerba'
    
    y.imshow(orig, cmap ='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()