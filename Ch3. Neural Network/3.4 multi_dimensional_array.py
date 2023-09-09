import numpy as np

####################################################################
## 1차원 배열 ##
A = np.array([1, 2, 3, 4])
print("A = ", A)
## 배열 차원 확인 ##
print("A dimension = ", np.ndim(A))
## 배열 형상 확인 ##
print("A shape = ", A.shape)
print("A shape [0] = ", A.shape[0])
####################################################################
## 2차원 배열 ##
B = np.array([[1, 2],[3, 4], [5, 6]])
print("B = ", B)
## 배열 차원 확인 ##
print("B dimension = ", np.ndim(B))
## 배열 형상 확인 ##
print("B shape = ", B.shape)
####################################################################