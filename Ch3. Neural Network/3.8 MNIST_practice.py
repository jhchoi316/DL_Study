import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

####################################################################
## load dataset ##
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

print("x_train shape = ", x_train.shape)
print("t_train shape = ", t_train.shape)
print("x_test shape = ", x_test.shape)
print("t_test shape = ", t_test.shape)
####################################################################
## check dataset ##
from PIL import Image
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
    
img = x_train[0]
label = t_train[0]
print("label = ", label)

print("image shape = ", img.shape)
img = img.reshape(28, 28)
print("imgage reshape = ", img.shape)

img_show(img)
####################################################################
## 함수 정의 ##
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

## pickle 파일에 저장된 '학습된 가중치 매개변수'를 읽음 ##
def init_network():
    with open("/Users/jaehunchoi/Desktop/JAE/Study/DL_Study/Ch3. Neural Network/dataset/sample_weight.pkl", 'rb') as f: 
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y
####################################################################
## 정확도 평가 ##
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
    
####################################################################
## 신경망 각 층의 가중치 형상 확인 ##
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print("x shape = ", x.shape)
print("x[0] shape = ", x[0].shape)
print("W1 shape = ", W1.shape)
print("W2 shape = ", W2.shape)
print("W3 shape = ", W3.shape)
####################################################################
## 배치 처리 ##
x, t = get_data()
network= init_network()

batch_size = 100
accuracy_cnt = 0


for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
####################################################################
## range의 예시 ##
print("range(0,10) = ", list(range(0, 10)))
print("range(0,10,3) = ", list(range(0,10,3)))
####################################################################
## axis = 1 의 예시 ##
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6],
             [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis = 1)
print("y = ", y)

t = np.array([1, 2, 0, 0])
print("y==t = ", y==t)
print("np.sum(y==t) = ", np.sum(y==t))
####################################################################