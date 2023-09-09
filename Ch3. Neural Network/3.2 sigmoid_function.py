import numpy as np
import matplotlib.pylab as plt

####################################################################
## 시그모이드 함수 구현 ##
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print("x = ", x)
y = sigmoid_function(x)
print("y = ", y)

t = np.array([1.0, 2.0, 3.0])
print("1.0 + t = ", 1.0 + t)
print("1.0 / t = ", 1.0 / t)
####################################################################
## 시그모이드 함수 그래프 ##
x = np.arange(-5.0 , 5.0, 0.1)
y = sigmoid_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
####################################################################
