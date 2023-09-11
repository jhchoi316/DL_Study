import numpy as np

####################################################################
## 1층의 첫 번째 뉴런 구현 ##
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print("W1 shape = ", W1.shape)
print("X shape = ", X.shape)
print("B1 shape = ", B1.shape)

A1 = np.dot(X, W1) + B1
print("A1 = ", A1)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

Z1 = sigmoid_function(A1)
print("Z1 = ", Z1)
####################################################################
## 1층에서 2층으로의 신호 전달 ##
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print("Z1.shape = ", Z1.shape)
print("W2 shape = ", W2.shape)
print("B2 shape = ", B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid_function(A2)
print("Z2 = ", Z2)
####################################################################
## 2층에서 출력층으로의 신호 전달 ##
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print("Y = ", Y)
####################################################################
## 구현 총 정리 ##
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print("y = ", y)
####################################################################
