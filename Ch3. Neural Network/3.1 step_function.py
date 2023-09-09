import numpy as np
import matplotlib.pylab as plt

####################################################################
## 계단 함수 구현 ##
def step_function(x):
    ## 실수만 받아드리는 코드 ##
    # if x > 0:
    #     return 1
    # else:
    #     return 0
    
    ## numpy 배열을 위한 코드 ##
    y = x > 0
    return y.astype(np.int32)

x = np.array([-1.0, 1.0, 2.0])
print("x = ", x)
## bool ##
y = x > 0
print("y = ", y)
## int ###
y = y.astype(np.int32)
print("y = ", y)
####################################################################
## 계단 함수 그래프 ##
def step_function_graph(x):
    return np.array(x > 0, dtype = np.int32)

## -5.0 ~ 5.0을 0.1 간격으로 ##
x = np.arange(-5.0, 5.0, 0.1)
y = step_function_graph(x)

plt.plot(x, y)
## y축 범위 지정 ##
plt.ylim(-0.1, 1.1)
plt.show()
####################################################################
