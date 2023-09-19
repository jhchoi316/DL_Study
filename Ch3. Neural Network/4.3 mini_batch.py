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

