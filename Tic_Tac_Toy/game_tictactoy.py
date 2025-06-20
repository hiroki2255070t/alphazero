import random
import math

class State:
    def __init__(self, pieces=None, enemy_pieces=None):
        # 石の配置
        if(pieces != None):
            self.pieces = pieces
        else:
            self.pieces = [0] * 9
        if(enemy_pieces != None):
            self.enemy_pieces = enemy_pieces
        else:
            self.enemy_pieces = [0] * 9

    # 石の数の取得
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if(i == 1):
                count += 1
        return count
    
    # 負けがどうか
    def is_lose(self):
        def is_comp(x, y, dx, dy):
            for k in range(3):
                if(y < 0 or y > 2 or x < 0 or x > 2 or self.enemy_pieces[x+y*3] == 0):
                    return False
                x, y = x+dx, y+dy
            return True
        
        if(is_comp(0, 0, 1, 1) or is_comp(0, 2, 1, -1)):
            return True
        for i in range(3):
            if(is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1)):
                return True
        return False
    
    # 引き分けかどうか
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 9
    
    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()
    
    # 次の状態の取得
    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)
    
    # 合法手リストの取得
    def legal_actions(self):
        actions = []
        for i in range(9):
            if(self.pieces[i] + self.enemy_pieces[i] == 0):
                actions.append(i)
        return actions
    
    # 先手かどうか
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)
    
    # 文字列表示
    def __str__(self):
        if(self.is_first_player()):
            ox = ('○', '●')
        else:
            ox = ('●', '○')
        str = ''
        for i in range(9):
            if(self.pieces[i] == 1):
                str += ox[0]
            elif(self.enemy_pieces[i] == 1):
                str += ox[1]
            else:
                str += '□'
            if(i % 3 == 2):
                str += '\n'
        return str
    

def random_action(state: State):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]


# プレイアウト
def playout(state: State):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    
    return -playout(state.next(random_action(state)))

# 最大値のインデックスを返す
def argmax(collection, key=None):
    return collection.index(max(collection))


def alpha_beta(state: State, alpha, beta):
    if(state.is_lose()):
        return -1
    if(state.is_draw()):
        return 0
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if(score > alpha):
            alpha = score
        if(alpha >= beta):
            return alpha

    return alpha        

def alpha_beta_action(state: State):
    best_action = 0
    alpha = -float('inf')
    str = ['','']
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if(score > alpha):
            best_action = action
            alpha = score
        
        str[0] = '{}{:2d},'.format(str[0], action)
        str[1] = '{}{:2d},'.format(str[1], score)
    print('action:', str[0], '\nscore:', str[1], '\n')

    return best_action


def mcts_action(state: State):
    # モンテカルロ木探索のノードの定義
    class Node:
        def __init__(self, state: State):
            self.state = state
            self.w = 0
            self.n = 1
            self.child_nodes = None

        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0

                self.w += value
                self.n += 1
                return value
            
            # 子ノードが存在しない場合
            if not self.child_nodes:
                value = playout(self.state)

                self.w += value
                self.n += 1

                if self.n == 10:
                    self.expand()
                return value
            
            else:
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1
                return value
            

        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))


        def next_child_node(self):
            t = 0
            for child_node in self.child_nodes:
                t += child_node.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n + (2*math.log(t)/child_node.n)**0.5)
            
            return self.child_nodes[argmax(ucb1_values)]
        
    # 現在の局面のノードの作成
    root_node = Node(state)
    root_node.expand()

    # 100回のシミュレーションを実行
    for _ in range(100):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for child_node in root_node.child_nodes:
        n_list.append(child_node.n)
    return legal_actions[argmax(n_list)]

# 動作確認
if __name__ == '__main__':
    state = State()

    while True:
        if state.is_done():
            break

        action = random_action(state)
        state = state.next(action)

        print(state)
        print()