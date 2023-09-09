import numpy as np
import matplotlib.pyplot as plt

####################################################################
## 데이터 준비 ##
## 0~6까지 0.1 간격으로 생성 ##
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

## 그래프 그리기 ##
plt.plot(x, y1, label = "sin")
plt.plot(x, y2, linestyle="--", label = "cos")
## x축 이름 ##
plt.xlabel("x")
## y축 이름 ##
plt.ylabel("y")
## 그래프 제목 ##
plt.title('sin & cos')
plt.legend()
plt.show()
####################################################################
from matplotlib.image import imread

img = imread('./img/wolf-dog.jpeg')
plt.imshow(img)
plt.show()
####################################################################