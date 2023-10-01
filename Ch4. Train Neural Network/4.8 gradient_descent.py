import numpy as np

####################################################################
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val +h
        fxh1 = f(x)
        
        x[idx] = tmp_val -h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad

## 경사하강법 구현 ##
# f: 최적화하려는 함수, init_x: 초기값, lr: 학습률, step_num: 반복 횟수 #
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print("gradient_descent(function_2, init_x=init_x, lr = 0.1, step_num=100) = ", gradient_descent(function_2, init_x=init_x, lr = 0.1, step_num=100))
####################################################################
import numpy as np
import matplotlib.pylab as plt

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    
lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
####################################################################
## 학습률에 다른 결과 ##
# 학습률이 클 때 #
init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr = 10.0, step_num=100)
print("gradient_descent(function_2, init_x=init_x, lr = 10.0, step_num=100) = ",result)

# 학습률이 작을 때 #
init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr = 1e-10, step_num=100)
print("gradient_descent(function_2, init_x=init_x, lr = 1e-10, step_num=100) = ", result)
####################################################################
