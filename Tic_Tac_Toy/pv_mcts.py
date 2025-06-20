from game_tictactoy import State
from Tic_Tac_Toy.dual_network import DN_INPUT_SHAPE
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D
from keras.models import Model
from pathlib import Path
import numpy as np
from tensorflow.python.keras.models import load_model


PV_EVALUATE_COUNT = 50

# 推論
def predict(model: Model, state: State):
    # 推論のための入力データの形状の変換
    a, b, c = DN_INPUT_SHAPE
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)

    # 推論
    y = model.predict(x, batch_size=1)

    # 方策の取得
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1

    # 価値の取得
    value = y[1][0][0]
    return policies, value


# ノードの試行回数のリストをスコアのリストに変換
def nodes_to_scores(nodes):
    scores = []
    for node in nodes:
        scores.append(node.n)
    return scores

# モンテカルロ木探索で行動選択
def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state: State):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# ボルツマン分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

# モンテカルロ木探索のスコアの取得
def pv_mcts_scores(model: Model, state: State, temperature):
    # ノードの定義
    class Node:
        def __init__(self, state: State, p):
            self.state = state
            self.p = p
            self.w = 0
            self.n = 0
            self.child_nodes = None

        # 局面の価値の計算
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 価値の取得
                value = -1 if self.state.is_lose() else 0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value
            
            # 子ノードが存在しない場合
            if not self.child_nodes:
                # ニューラルネットワークの推論で方策と価値を取得
                policies, value = predict(model, self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
                return value
            
            # 子ノードが存在する場合
            else:
                # アーク評価値が最大の子ノードの評価を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value
        
        # アーク評価値が最大の子ノードを取得
        def next_child_node(self):
            # アーク評価値の計算
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0)
                                   + C_PUCT * child_node.p * sqrt(t) / (1 +child_node.n))
            
            # アーク評価値が最大の子ノードを返す
            return self.child_nodes[np.argmax(pucb_values)]
        
    # 現在の局面のノードの作成
    root_node = Node(state, 0)

    # 複数回の評価を実行
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    
    return scores


# 動作確認
if __name__ == '__main__':
    path = sorted(Path('./model').glob('*.h5'))[-1]
    model = load_model(str(path), custom_objects={'BatchNormalization': BatchNormalization})

    state = State()
    next_action = pv_mcts_action(model, 1.0)

    while True:
        if state.is_done():
            break

        action = next_action(state)
        state = state.next(action)

        print(state)