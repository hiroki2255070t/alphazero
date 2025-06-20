from game_tictactoy import State
from pv_mcts import pv_mcts_action
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import backend as K
from pathlib import Path
from shutil import copy
import numpy as np


EN_GAME_COUNT = 10
EN_TEMPERATURE = 1.0

# 先手プレイヤーのポイント
def first_player_point(ended_state: State):
    # 1:先手勝利 0:先手敗北 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# 1ゲームの実行
def play(next_actions):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)

# ベストプレイヤーの交代
def update_best_player():
    copy('./model/latest_model', './model/best_model')
    print('Change BestPlayer')

# ネットワークの評価
def evaluate_network():
    # 最新プレイヤーモデルの読み込み
    model_latest = load_model('./model/latest_model')
    
    # 現在のベストプレイヤーモデルの読み込み
    model_best = load_model('./model/best_model')

    # PV MCTSで行動選択を行う関数の生成
    next_action_latest = pv_mcts_action(model_latest, EN_TEMPERATURE)
    next_action_best = pv_mcts_action(model_best, EN_TEMPERATURE)
    next_actions = (next_action_latest, next_action_best)

    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EN_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)

    # モデルの破棄
    K.clear_session()
    del model_latest
    del model_best

    # ベストプレイヤーモデルの交代
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False
    
# 動作確認
if __name__ == '__main__':
    evaluate_network()