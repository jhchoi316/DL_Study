import numpy as np
import matplotlib.pylab as plt

####################################################################
## ReLU 함수 구현 ##
def relu_function(x):
    ## maximum 함수는 두 입력 중 큰 값을 선택해 반환 ##
    return np.maximum(0, x)

x = np.array([-1.0, 1.0, 2.0])
print("x = ", x)
y = relu_function(x)
print("y = ", y)
####################################################################
## ReLU 함수 그래프 ##
x = np.arange(-5.0 , 5.0, 0.1)
y = relu_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
####################################################################



