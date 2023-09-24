import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

####################################################################
## 교차 엔트로피 오차 구현 ##
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

    # 정답 레이블이 one hot encoding이 아닌 숫자 레이블일 경우 ##
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
####################################################################