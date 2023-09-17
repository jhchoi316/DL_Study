import numpy as np

####################################################################
## 소프트맥스 함수 구현 ##
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print("exp_a = ", exp_a)

sum_exp_a = np.sum(exp_a)
print("sum_exp_a = ", sum_exp_a)

y = exp_a / sum_exp_a
print(" y = ", y)
####################################################################
## 소프트맥스 함수 논리 흐름 구현 ##
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
####################################################################
## 개선된 소프트맥스 함수(overflow 방지) ##
a = np.array([1010, 1000, 990])
print("overflow softmax version = ", np.exp(a) / np.sum(np.exp(a)))

c = np.max(a)
print("a-c = ", a-c)

print("no overflow softmax version = ", np.exp(a-c) / np.sum(np.exp(a-c)))
####################################################################
## 개선된 소프트맥스 함수 논리 흐름 구현 ##
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
####################################################################
