import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

####################################################################
## 미니 배치 학습 ##
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
    
print("x_train shape = ", x_train.shape)
print("x_test shape = ",x_test.shape)
####################################################################
## 훈련 데이터에서 무작위 10장만 뺴내기 ##
train_size = x_train.shape[0]
batch_size = 10
# 지정한 범위의 수 중에서, 무작위로 원하는 개수만 꺼낼 수 있음 ##
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print("x_batch = ", x_batch)
print("t_batch = ", t_batch)