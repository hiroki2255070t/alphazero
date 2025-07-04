from Tic_Tac_Toy.dual_network import DN_INPUT_SHAPE
import tensorflow
from tensorflow import keras
from keras.callbacks import LearningRateScheduler, LambdaCallback
from keras.models import load_model
from keras import backend as K
from pathlib import Path
import numpy as np
import pickle


RN_EPOCHS = 100

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data').glob('*history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)
    
# デュアルネットワークの学習
def train_network():
    # 学習データの読み込み
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    # 学習のための入力データの形状の変換
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # ベストプレイヤーのモデルの読み込み
    model = load_model('./model/best_model')

    # モデルのコンパイル
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    # 学習率
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    # 出力
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch,logs:
        print('\rTrain {}/{}'.format(epoch+1, RN_EPOCHS), end=''))
    
    # 学習の実行
    model.fit(xs, [y_policies, y_values], batch_size=128, epochs=RN_EPOCHS,
              verbose=0, callbacks=[lr_decay, print_callback])
    print('')

    # 最新プレイヤーのモデルの保存
    model.save('./model/latest_model')

    # モデルの破棄
    K.clear_session()
    del model