#mnistのダウンロード。読み込みなど
import urllib.request #URLの読み込み
import gzip #ファイルの圧縮、展開
import pickle #複数オブジェクトを1つに保存
import os #os関連
import numpy as np #数列処理など

url_base = 'http://yann.lecun.com/exdb/mnist/'#mnistのURL
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}#各データセットの辞書

#mnistデータセットを保存するディレクトリを用意
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

#データセットの情報
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _download(file_name):#ファイルのダウンロードを行う関数
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):#既に対象のファイルが存在する場合は何もしない
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

def download_mnist():#mnistのダウンロードを行う関数
    for v in key_file.values():
       _download(v)

def _load_label(file_name):#ラベルの読み込み関数
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def _load_img(file_name):#画像の読み込み関数
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)#画像を配列に変換
    data = data.reshape(-1, img_size)#画像の配列を1次元に変換
    print("Done")

    return data

def _convert_numpy():#ダウンロードしたデータセットを配列に変換する関数
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

def init_mnist():#mnistのダウンロード、配列への変換、保存を行う関数
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):#ラベルデータをone-hot型に変換
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

#mnist読み込み関数
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):#mnistファイルがダウンロードされていない場合
        init_mnist()

    with open(save_file, 'rb') as f:#mnistファイルの読み込み
        dataset = pickle.load(f)

    if normalize:#正規化有効の場合
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:#one-hot有効の場合
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:#1次元配列が有効ではない場合画像データを28*28にする
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()

