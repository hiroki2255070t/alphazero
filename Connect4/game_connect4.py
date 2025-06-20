import random
import math

class State:
    # ゲームの状態の初期化
    def __init__(self, pieces=None, enemy_pieces=None):
        self.pieces = pieces if pieces != None else [0] * 42
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * 42
    
    # 負け判定
    def is_lose(self):
        # 4並び判定
        def is_comp(x, y, dx, dy):
            for i in range(4):
                if x < 0 or x > 6 or y < 0 or y > 5 or self.enemy_pieces[x + 7*y] == 0:
                    return False
                x, y = x+dx, y+dy
            return True
        
        top_position = self.top_positions()
        for j in range(6):
            for i in range(7):
                if is_comp(i, j, 1, 0) or is_comp(i, j, 1, 1) or \
                    is_comp(i, j, 1, -1) or is_comp(i, j, 0, 1):
                    return True  
        return False
    
    # 引き分け判定
    def is_draw(self):
        return sum(self.pieces) + sum(self.enemy_pieces) == 42

    # 空いているマスの一番上の位置を返す
    def top_positions(self):
        pieces = [x + y for x, y in zip(self.pieces, self.enemy_pieces)]
        top_positions = []
        for x in range(7):
            y = 0
            for i in range(6):
                if pieces[x + 7*y] == 0:
                    break
                y += 1
            top_positions.append(y)
        return top_positions
    
    # ゲーム終了判定
    def is_done(self):
        return self.is_lose() or self.is_draw()
    
    # 次の状態の取得
    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action + 7 * self.top_positions()[action]] = 1
        return State(self.enemy_pieces, pieces)
    
    # 合法手のリストを取得
    def legal_actions(self):
        legal_actions = []
        top_positions = self.top_positions()
        for x in range(7):
            if top_positions[x] != 6:
                legal_actions.append(x)
        return legal_actions
    
    # 先手かどうか判定
    def is_first_player(self):
        return sum(self.pieces) == sum(self.enemy_pieces)
    
    # 画面に盤面を表示
    def __str__(self):
        if self.is_first_player():
            ox = ('○ ', '● ')
        else:
            ox = ('● ', '○ ')
        str = ' '
        for y in range(5, -1, -1):
            for x in range(7):
                if(self.pieces[x + 7*y] == 1):
                    str += ox[0]
                elif(self.enemy_pieces[x + 7*y] == 1):
                    str += ox[1]
                else:
                    str += '□ '
            str += '\n '
        str += ('0 1 2 3 4 5 6\n')
        return str