import numpy as np
import matplotlib.pyplot as plt

####################################################################
## 함수 미분 나쁜 예 ##
def numerical_diff(f, x):
    h = 1e-50
    return(f(x+h) - f(x)) / h
####################################################################
## 함수 미분 좋은 예 ##
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
####################################################################
## 수치 미분의 예 ##
def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.show()

print("numberical_diff = ", numerical_diff(function_1, 5))
print("numberical_diff = ", numerical_diff(function_1, 10))
####################################################################
