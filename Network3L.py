# 3層ニューラルネットワークの実装
#ライブラリと外部ファイルから関数のインポート
from functions import *
import numpy as np

#ネットワークの実装
class Three_Layer_Net:
    name = 'Three_Layer_Net'
    keys = ('W1', 'b1', 'W2', 'b2')
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #input_size:入力層ニューロンの数。hidden_size:隠れ層ニューロンの数。output_size:出力層ニューロンの数。weight_init_std:重みの初期値を小さくするパラメータ
        # 重みWとバイアスbの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    #ネットワークの推論処理
    def predict(self, x):#x:ネットワークへの入力
        #ネットワークのパラメータの読み込み
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        #推論処理
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
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
        return grads
        
    #重みに対する勾配を計算する関数(高速版) x:入力データ, t:教師データ
    def gradient(self, x, t):
        #パラメータの読み込み
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        batch_num = x.shape[0]
        #誤差逆伝播法による勾配の計算
        # 順伝播
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        # 逆伝播
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        #各パラメータに対して勾配を計算して辞書にして返す
        return grads
