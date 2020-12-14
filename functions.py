#数学関数の保管場所
import numpy as np
#シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#ReLu関数
def relu(x):
    return np.maximum(0,x)
#シグモイド勾配
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
#Relu勾配
def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad
#ソフトマックス関数
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
#二乗和誤差
def mean_squarted_error(y,t):
    return 0.5*np.sum((y-t)**2)
#交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size
#勾配の計算
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
    return grad