import numpy as np
import random
import os
import copy
from tensorflow.keras.models import load_model
from nn import model_path
#from cnn import model_path
import function as fc
from function import x_len, y_len


epochs = 1
change_level = 3


#分類に使う配列
class_arr2 = np.array([[0,1,0],[1,0,1],[0,1,0]])
class_arr3 = np.array([[1,1,1],[0,0,0],[0,0,0]])
class_arr4 = np.array([[0,0,0],[0,0,0],[1,1,1]])
class_arr5 = np.array([[0,0,0],[1,0,1],[0,0,0]])

#初期個体生成
def create_first_design(mode):
    design = np.zeros((y_len,x_len))
    #最初の形を決める
    if mode == 0:
        design = fc.design()
    if mode == 1:
        #design[:, 13] = 1
        design[0:7,5:18] = 2
        design[0:7,4] = 1
        design[0:7,18] = 1
        design[0,5:18] = 1
    return design

#近傍解生成
def create_neighbor(design):
    #1で囲む
    design = np.insert(design, y_len, 1, axis=0)
    design = np.insert(design, 0, 1, axis=0)
    design = np.insert(design, x_len, 1, axis=1)
    design = np.insert(design, 0, 1, axis=1)

    pattern = 1
    neighbor = copy.copy(design)
    for cl in range (change_level):
        #変更箇所選出
        if pattern > 1:
            next = np.array(list(zip(*np.where(change_area==1))))
            loop = True
            for n in range(next.shape[0]):
                cp_x = cp_x -1 + next[n][1]
                cp_y = cp_y -1 + next[n][0]
                if 0<cp_x<x_len+1 and 0<cp_y<x_len+1:
                    loop = False
                    break
            if loop:
                pattern = 1
        if pattern == 1:
            #境界部抽出
            groove = np.array(list(zip(*np.where(design[1:y_len+1,1:x_len+1]==1))))+1
            #変更箇所選定
            n = random.randrange(0,len(groove))
            cp_x = groove[n][1]
            cp_y = groove[n][0]
        

        
        change_area = neighbor[cp_y-1:cp_y+2, cp_x-1:cp_x+2]

        #変更
        print(cp_x,cp_y,cl)
        print(change_area)
        _change_area = np.where(change_area>1, 0, change_area)
        pattern = np.sum(_change_area)
        print("pattern",pattern)
        subpattern = np.sum(class_arr2*_change_area)
        print("subpattern", subpattern)
        #分岐
        if pattern < 3:
            change_area[1][1] = 2
        elif pattern == 3:
            if subpattern == 0:#BF
                #F
                if np.all(_change_area == _change_area.T):
                    dise = random.randint(0,1)
                    if _change_area[0][0] == 1:
                        if dise == 0:
                            change_area[0][1] = 1
                            change_area[1][2] = 1
                            change_area[1][1] = change_area[1][0]
                        elif dise == 1:
                            change_area[1][0] = 1
                            change_area[2][1] = 1
                            change_area[1][1] = change_area[0][1]
                    elif _change_area[0][0] == 0:
                        if dise == 0:
                            change_area[1][2] = 1
                            change_area[2][1] = 1
                            change_area[1][1] = change_area[0][1]
                        elif dise == 1:
                            change_area[0][1] = 1
                            change_area[1][0] = 1
                            change_area[1][1] = change_area[1][2]
                else:
                    #B
                    if _change_area[0][0] == 1:
                        change_area[1][1] = change_area[2][2]
                        if _change_area[0][2] == 1:
                            change_area[0][1] = 1
                        elif _change_area[2][0] == 1:
                            change_area[1][0] = 1
                    elif _change_area[2][2] == 1:
                        change_area[1][1] = change_area[0][0]
                        if _change_area[0][2] == 1:
                            change_area[1][2] = 1
                        elif _change_area[2][0] == 1:
                            change_area[2][1] = 1
            elif subpattern == 1:#AD
                #D
                if np.sum(class_arr3*_change_area)==1 and np.sum(class_arr4*_change_area)==1:
                    if np.sum(class_arr3.T*_change_area) == 1:
                        change_area[1][0] = 1
                        change_area[1][1] = change_area[1][2]
                    elif np.sum(class_arr4.T*_change_area) == 1:
                        change_area[1][2] = 1
                        change_area[1][1] = change_area[1][0]
                elif np.sum(class_arr3.T*_change_area)==1 and np.sum(class_arr4.T*_change_area)==1:
                    if np.sum(class_arr3*_change_area) == 1:
                        change_area[0][1] = 1
                        change_area[1][1] = change_area[2][1]
                    elif np.sum(class_arr4*_change_area) == 1:
                        change_area[2][1] = 1
                        change_area[1][1] = change_area[0][1]
                #A
                else:
                    change_area[1][1] = 2
            elif subpattern == 2:#CE
                #E
                dise = random.randint(0,1)
                if np.sum(class_arr5*_change_area) == 2:
                    if dise == 0:
                        change_area[0][1] = 1
                        change_area[1][1] = change_area[2][1]
                    elif dise == 1:
                        change_area[2][1] = 1
                        change_area[1][1] = change_area[0][1] 
                elif np.sum(class_arr5.T*_change_area) == 2:
                    if dise == 0:
                        change_area[1][0] = 1
                        change_area[1][1] = change_area[1][2]
                    elif dise == 1:
                        change_area[1][2] = 1
                        change_area[1][1] = change_area[1][0]
                #C
                elif np.sum(class_arr3*_change_area) == 1:
                    change_area[1][1] == change_area[2][1]
                elif np.sum(class_arr4*_change_area) == 1:
                    change_area[1][1] = change_area[0][1]
        elif pattern == 4:
            if subpattern == 0:#G　未実装
                print("===4-G===")
                if _change_area[0][0] == 0:
                    _change_area[1:,1:] = 1
                    _change_area[2][2] = 0
                elif _change_area[0][2] == 0:
                    _change_area[1:,:2] = 1
                    _change_area[2][0] = 0
                elif _change_area[2][0] == 0:
                    _change_area[:2,1:] = 1
                    _change_area[0][2] = 0
                elif _change_area[2][2] == 0:
                    _change_area[:2,:2] = 1
                    _change_area[0][0] = 0
            elif subpattern == 1:#ADFH
                if np.sum(class_arr3*_change_area) >= 2:
                    change_area[1][1] = change_area[2][1]
                    if _change_area[2][0] == 1:
                        change_area[1][0] = 1
                    elif _change_area[2][2] == 1:
                        change_area[1][2] = 1
                    elif np.sum(class_arr5*_change_area) == 1:
                        change_area[0][1] = 1
                    elif _change_area[2][1] == 1:
                        print("===4-H===")
                elif np.sum(class_arr4*_change_area) >= 2:
                    change_area[1][1] = change_area[0][1]
                    if _change_area[0][0] == 1:
                        change_area[1][0] = 1
                    elif _change_area[0][2] == 1:
                        change_area[1][2] = 1
                    elif np.sum(class_arr5*_change_area) == 1:
                        change_area[2][1] = 1
                    elif _change_area[0][1] == 1:
                        print("===4-H===")
                elif np.sum(class_arr3.T*_change_area) >= 2:
                    change_area[1][1] = change_area[1][2]
                    if _change_area[0][2] == 1:
                        change_area[0][1] = 1
                    elif _change_area[2][2] == 1:
                        change_area[2][1] = 1
                    elif _change_area[1][2] == 1:
                        print("===4-H===")
                elif np.sum(class_arr4.T*_change_area) >= 2:
                    change_area[1][1] = change_area[1][0]
                    if _change_area[0][0] == 1:
                        change_area[0][1] = 1
                    elif _change_area[2][0] == 1:
                        change_area[2][1] = 1
                    elif _change_area[1][0] == 1:
                        print("===4-H===")
            elif subpattern == 2:#BCEJ
                #E
                if np.sum(class_arr5*_change_area) == 2:
                    if np.sum(class_arr3*_change_area) == 1:
                        change_area[2][1] = 1
                        change_area[1][1] = change_area[0][1]
                    elif np.sum(class_arr4*_change_area) == 1:
                        change_area[0][1] = 1
                        change_area[1][1] = change_area[2][1]
                elif np.sum(class_arr5.T*_change_area) == 2:
                    if np.sum(class_arr3.T*_change_area) == 1:
                        change_area[1][2] = 1
                        change_area[1][1] = change_area[1][0]
                    elif np.sum(class_arr4.T*_change_area) == 1:
                        change_area[1][0] = 1
                        change_area[1][1] = change_area[1][2]
                #BC
                elif np.sum(class_arr3*_change_area)==0 or np.sum(class_arr3.T*_change_area)==0:
                    change_area[1][1] = change_area[0][0]
                elif np.sum(class_arr4*_change_area)==0 or np.sum(class_arr4.T*_change_area)==0:
                    change_area[1][1] = change_area[2][2]
                #J
                else:
                    dise = random.randint(0,1)
                    if _change_area[0][0] == 1:
                        if dise == 0:
                            change_area[1][0] = 1
                            change_area[1][1] = change_area[0][1]
                        elif dise == 1:
                            change_area[0][1] = 1
                            change_area[1][1] = change_area[1][0]
                    elif _change_area[0][2] == 1:
                        if dise == 0:
                            change_area[0][1] = 1
                            change_area[1][1] = change_area[1][2]
                        elif dise == 1:
                            change_area[1][2] = 1
                            change_area[1][1] = change_area[0][1]
                    elif _change_area[2][0] == 1:
                        if dise == 0:
                            change_area[1][0] = 1
                            change_area[1][1] = change_area[2][1]
                        elif dise == 1:
                            change_area[2][1] = 1
                            change_area[1][1] = change_area[1][0]
                    elif _change_area[2][2] == 1:
                        if dise == 0:
                            change_area[1][2] = 1
                            change_area[1][1] = change_area[2][1]
                        elif dise == 0:
                            change_area[2][1] = 1
                            change_area[1][1] = change_area[1][2]
            elif subpattern == 3:#I:
                if np.sum(class_arr5*_change_area) == 2:
                    if _change_area[0][1] == 1:
                        change_area[1][1] = change_area[2][1]
                    elif _change_area[2][1] == 1:
                        change_area[1][1] = change_area[0][1]
                elif np.sum(class_arr5.T*_change_area) == 2:
                    if _change_area[1][0] == 1:
                        change_area[1][1] = change_area[1][2]
                    elif _change_area[1][2] == 1:
                        change_area[1][1] = change_area[1][0]
        elif pattern == 5:
            if subpattern == 0:#L 未実装
                print("===5-L===")
            elif subpattern == 1:#BJ J未実装
                #B
                if np.sum(class_arr3*_change_area)==3 or np.sum(class_arr4*_change_area)==3:
                    if np.sum(class_arr3.T*_change_area) == 2:
                        change_area[1][0] = 1
                        change_area[1][1] = change_area[1][2]
                    elif np.sum(class_arr4.T*_change_area) == 2:
                        change_area[1][2] = 1
                        change_area[1][1] = change_area[1][0]
                elif np.sum(class_arr3.T*_change_area)==3 or np.sum(class_arr4.T*_change_area)==3:
                    if np.sum(class_arr3*_change_area) == 2:
                        change_area[0][1] = 1
                        change_area[1][1] = change_area[2][1]
                    elif np.sum(class_arr4*_change_area) == 2:
                        change_area[2][1] = 1
                        change_area[1][1] = change_area[0][1]
                #J
                else:
                    print("===5-J===")
            elif subpattern == 2:#ACEFHIK　CIKのみ実装
                #CIK
                if np.sum(class_arr5*_change_area) == 2:
                    #I
                    if np.sum(class_arr3*_change_area) == 2:
                        change_area[2][1] = 1
                        change_area[1][1] = change_area[0][1]
                    elif np.sum(class_arr4*_change_area) == 2:
                        change_area[0][1] = 1
                        change_area[1][1] = change_area[2][1]
                    #CK
                    else:
                        dise = random.randint(0,1)
                        if dise == 0:
                            change_area[0][1] = 1
                            change_area[1][1] = change_area[2][1]
                        elif dise == 1:
                            change_area[2][1] = 1
                            change_area[1][1] = change_area[0][1]
                elif np.sum(class_arr5.T*_change_area) == 2:
                    #I
                    if np.sum(class_arr3.T*_change_area) == 2:
                        change_area[1][2] = 1
                        change_area[1][1] = change_area[1][0]
                    elif np.sum(class_arr4.T*_change_area) == 2:
                        change_area[1][0] = 1
                        change_area[1][1] = change_area[1][2]
                    #CK
                    else:
                        dise = random.randint(0,1)
                        if dise == 0:
                            change_area[1][0] = 1
                            change_area[1][1] = change_area[1][2]
                        elif dise == 1:
                            change_area[1][2] = 1
                            change_area[1][1] = change_area[1][0]
            #elif subpattern == 3:#DG
        if pattern > 5:#else:
            print(pattern,"が出ました")
            cl -= 1

        print(change_area,cl)
        neighbor[cp_y-1:cp_y+2, cp_x-1:cp_x+2] = change_area

    #穴埋め
    groove = np.array(list(zip(*np.where(design[1:y_len+1,1:x_len+1]==1))))+1
    for n in groove:
        area = neighbor[n[0]-1:n[0]+2, n[1]-1:n[1]+2]
        if np.prod(area) > 0:
            area[1][1] = 2
    return neighbor[1:y_len+1, 1:x_len+1]


def main():
    #初期個体生成
    design = create_first_design(0)
    #評価
    model = load_model(model_path)
    eval = model.predict(design)

    for i in range(epochs):
        #近傍解生成
        #評価
        #入れ替え
        print(i)
    
    print(design)


if __name__ == '__main__':
    design = create_first_design(0)
    neighbor = create_neighbor(design)
    print(design)
    print(neighbor)
    print(np.all(design==neighbor))
    #neighbors = create_first_design()
    #print(neighbors)
    #path = '/data_result.txt'
    #data = neighbors[1:y_len+1, 1:x_len+1]
    #with open(os.getcwd()+path, mode='w') as f:
    #    for i in range(data.shape[0]):
    #        for n in data[i]:
    #            f.write(str(n)+" ")
    #        f.write('\n')
    #    f.write('\n')
    ##fc.write_data('/data_no.txt', neighbors[1:y_len+1, 1:x_len+1])