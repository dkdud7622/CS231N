import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class NearestNeighbor(object) :
    def __init__(self):
        pass

    def train(self, X, y) :
        self.Xtr = X
        self.ytr = y

    def predict(self, X) :
        num_test = X.shape[0]   # num of train data
        # 인풋인 ytr과 Ypred의 ypte을 일치시킨다.
        Ypred = np.zeros(num_test,dtype = self.ytr.dtype)
        count = 0
        for i in range (num_test) :
            distances = np.sum(np.abs(self.Xtr - X[i,:]),axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
            print(count+i)
        
        return Ypred



(Xtr, Ytr), (Xte, Yte) = tf.keras.datasets.cifar10.load_data()

#plot의 크기는 10*10
plt.rcParams['figure.figsize'] = (10.0,10.0)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#cifar10의 class이름
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(classes)
samples_per_class = 5

#y: class의 인덱스
#cls: class의 각 이름
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(Ytr == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        #plt 상에서 해당 data가 위치할 인덱스(column 인덱스)
        plt_idx = i * num_classes + y + 1
        #세로 7, 가로 10 사이즈 plot 내에 plt_idx 위치에 subplot 생성
        plt.subplot(samples_per_class, num_classes, plt_idx)
        #train data의 인덱스에 존재하는 이미지를 uint8 형식으로 출력
        plt.imshow(Xtr[idx].astype('uint8'))
        #plt의 좌표축은 무시
        plt.axis('off')
        #첫 줄일때, class 이름을 출력
        if i == 0:
            plt.title(cls)
#plt 출력
# plt.show()

#image dimension flat (become 1-dim)
Xtr_rows = Xtr.reshape(Xtr.shape[0],32*32*3)    #50000, 3072
Xte_rows = Xte.reshape(Xte.shape[0],32*32*3)    #10000, 3072

nn = NearestNeighbor()
nn.train(Xtr_rows,Ytr)

Yte_predict = nn.predict(Xte_rows)

Xval_rows = Xtr_rows[:1000,:]
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[:1000,:]
Ytr = Ytr[:1000]

validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

    # use a particular value of k and evaluation on validation data
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    # here we assume a modified NearestNeighbor class that can take a k as input
    Yval_predict = nn.predict(Xval_rows, k = k)
    acc = np.mean(Yval_predict == Yval)
    print('accuracy: ' + acc)

    # keep track of what works on the validation set
    validation_accuracies.append((k, acc))

print(validation_accuracies)