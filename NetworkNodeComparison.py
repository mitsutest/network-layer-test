#ネットワークのノード数による学習の変化を見る
#ライブラリと外部ファイルから関数のインポート
import numpy as np
import matplotlib.pylab as plt
from Mnist import load_mnist
from Network3L import Three_Layer_Net
#学習過程の保存をする関数
def Record(network, iter_per_epoch, test_acc_list, num):
    if i % iter_per_epoch == 0:
        #一定回数ごとに精度を計算してリストに保存。
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        #ネットワークの種類、学習データでの精度、テストデータでの精度を表示。
        print("Network, train acc, test acc | network" + str(num) + ", " + str(train_acc) + ", " + str(test_acc))
#学習データとテストデータの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#学習過程を保存するリスト
test_acc_list1 = []
test_acc_list2 = []
test_acc_list3 = []
#ハイパーパラメータの設定(イテレーションの回数、学習データのサイズ、バッチサイズ、学習率)
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
#1エポックごとの繰り返しの回数
iter_per_epoch = max(train_size / batch_size, 1)
#ネットワークの定義(同じネットワークで隠れ層のノード数を変化させる)
network1 = Three_Layer_Net(input_size=784, hidden_size=100, output_size=10)
network2 = Three_Layer_Net(input_size=784, hidden_size=50, output_size=10)
network3 = Three_Layer_Net(input_size=784, hidden_size=25, output_size=10)
#学習プロセス
for i in range(iter_num):
    #バッチの取得
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    #勾配の計算
    grad1 = network1.gradient(x_batch,t_batch)
    grad2 = network2.gradient(x_batch,t_batch)
    grad3 = network3.gradient(x_batch,t_batch)
    #パラメータの更新
    for key in network1.keys:
        network1.params[key] -= learning_rate * grad1[key]
    for key in network2.keys:
        network2.params[key] -= learning_rate * grad2[key]
    for key in network3.keys:
        network3.params[key] -= learning_rate * grad3[key]
    #学習経過の記録
    Record(network1,iter_per_epoch,test_acc_list1,1)
    Record(network2,iter_per_epoch,test_acc_list2,2)
    Record(network3,iter_per_epoch,test_acc_list3,3)
# グラフの描画:各ネットワークのテストデータでの精度をグラフに表示
x = np.arange(len(test_acc_list1))
plt.plot(x, test_acc_list1, label='network1')
plt.plot(x, test_acc_list2, label='network2', linestyle=':')
plt.plot(x, test_acc_list3, label='network3', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
