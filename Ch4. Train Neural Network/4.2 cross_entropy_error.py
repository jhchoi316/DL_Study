import numpy as np

####################################################################
def cross_entropy_error(y, t):
    # 델타는, 로그함수에 0을 입력하게되면 마이너스 무한대를 뜻하는 -inf가 되어 계산을 진행할 수 없기에 추가 #
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print("cross_entropy_error(y, t) = ", cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print("cross_entropy_error(y, t) = ", cross_entropy_error(np.array(y), np.array(t)))
####################################################################
