import numpy as np

####################################################################
## 행렬 곱 ##
A = np.array([[1, 2], [3, 4]])
print("A shape = ", A.shape)
B = np.array([[5, 6], [7, 8]])
print("B shape = ", B.shape)
## np.dot()은 행렬의 곱을 계산해주는 함수 ##
print("A x B = ", np.dot(A, B))

## A의 행과 B의 열의 형상이 같아야 곱이 가능 ##
A = np.array([[1, 2], [3, 4]])
print("A shape = ", A.shape)
B = np.array([[5, 6], [7, 8], [9, 10]])
print("B shape = ", B.shape)

if(A.shape[1] == B.shape[0]):
    print("A x B = ", np.dot(A, B))
else: print("A x B Shape Different!")
if (A.shape[0] == B.shape[1]):
    print("B x A = ", np.dot(B, A))
else: print("B x A Shape different!")
####################################################################
## 2차원과 1차원 배열 곱 ##
A = np.array([[1, 2], [3, 4], [5, 6]])
print("A shape = ", A.shape)
B = np.array([7, 8])
print("B.shape = ", B.shape)
print("A x B = ", np.dot(A, B))
####################################################################
## 신경망에서의 행렬 곱 ##
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print("X x W = ", Y)
####################################################################