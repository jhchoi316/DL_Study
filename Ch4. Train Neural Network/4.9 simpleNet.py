import sys, os
sys.path.append(os.pardir)
import numpy as np

####################################################################
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad
####################################################################
## simpleNet 구현 ##
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화
    # 예측을 수행하는 매서드 #
    def predict(self, x):
        return np.dot(x, self.W)

    # 손실 함수의 값을 구하는 매서드 #
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print("simpleNet = ", net.W)

## 입력 데이터 ##
x = np.array([0.6, 0.9])
p = net.predict(x)
print("net.predict(x) = ", p)
print("최댓값의 인덱스 = ", np.argmax(p))

## 레이블 ##
t = np.array([0, 0, 1])
l = net.loss(x, t)
print("net.loss(x,t) = ", l)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)

print("numerical_gradient(f, net.W) = ", dW)

## 람다식으로 표현 ##
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print("numerical_gradient(f, net.W) = ", dW)
####################################################################
