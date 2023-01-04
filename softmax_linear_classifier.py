import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

N = 100 #클래스당 포인트의 개수
D = 2   #차수
K = 3   #클래스 갯수
X = np.zeros((N*K,D))               # data matrix (300,2)
y = np.zeros(N*K, dtype='uint8')    # class labels (300, )

# print(X.shape, y.shape)

# 포인트를 랜덤하게 흩뿌리기 (매 실행 결과 달라짐)
for j in range(K) :
    ix = range(N*j, N*(j+1))
    # linsapce(start, stop, num) -> start와 stop 사이를 N개의 일정한 간격으로 나누어줌.
    r = np.linspace(0.0,1,N)                                 # radius
    t = np.linspace(j*4, (j+1)*4,N) + np.random.randn(N)*0.2 # theta

    #np.c_ : 두 개의 1차원 배열을 칼럼으로 세로로 붙여서 2차원 배열 만들기
    X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
    y[ix] = j

# data visualize
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral,alpha=0.9, edgecolors='black')
plt.show()


'''
Softmax Classifier
initialize parameters randomly
'''
W = 0.01 * np.random.randn(D,K) # (2, 3)
b = np.zeros((1,K))             # (1, 3)


# hyperparameters
step_size = 1e-0
reg = 1e-3  # regularization 강도


# gradient descent loop
num_examples = X.shape[0]
for i in range(200) :
    # 선형 분류기의 class score 계산
    score = np.dot(X,W)+b

    exp_score = np.exp(score)
    # probs의 각 행은 클래스 확률을 포함한다. (300*3) 정규화를 했기 때문에, 행의 합은 1이된다.
    probs = exp_score/ np.sum(exp_score,axis=1,keepdims=True) #(300,)
    
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples),y])    # (300,)
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss

    if i % 10 == 0 :
        print("iteration ",i,": loss ",loss)
    
    # 스코어의 그래디언트 계산하기.
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores/= num_examples


    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg*W # regularization gradient

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db


# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

h = 0.02
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, alpha=0.9, edgecolors='black')
plt.show()