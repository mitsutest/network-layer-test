# 4層ニューラルネットワークの実装
#ライブラリと外部ファイルから関数のインポート
from functions import *
import numpy as np

#ネットワークの実装
class Four_Layer_Net:
    name = 'Four_Layer_Net'
    keys = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, weight_init_std=0.01):
        #input_size:入力層ニューロンの数。hidden_size:隠れ層ニューロンの数。output_size:出力層ニューロンの数。weight_init_std:重みの初期値を小さくするパラメータ
        # 重みWとバイアスbの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden1_size)
        self.params['b1'] = np.zeros(hidden1_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden1_size, hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden2_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    #ネットワークの推論処理
    def predict(self, x):#x:ネットワークへの入力
        #ネットワークのパラメータの読み込み
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        #推論処理
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        return y
        
    #損失を計算する関数 x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        #ネットワークの推論処理の結果と教師データの交差エントロピー誤差を返す
        return cross_entropy_error(y, t)
    
    #認識精度を計算する関数
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        #推論を行って結果と教師データのラベルが一致した数を全体の数で割って精度を計算
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    #重みに対する勾配を計算する関数 x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        #各パラメータに対して勾配を計算して辞書にして返す
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        return grads
        
    #重みに対する勾配を計算する関数(高速版) x:入力データ, t:教師データ
    def gradient(self, x, t):
        #パラメータの読み込み
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}
        batch_num = x.shape[0]
        #誤差逆伝播法による勾配の計算
        # 順伝播
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        # 逆伝播
        dy = (y - t) / batch_num
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
        dz2 = np.dot(dy, W3.T)
        da2 = sigmoid_grad(a2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        dz1 = np.dot(dz2, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        #各パラメータに対して勾配を計算して辞書にして返す
        return grads
