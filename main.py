import numpy as np
import random
#from tensorflow.keras.models import load_model
#from nn import model_path
#from cnn import model_path
import design
from design import x_len, y_len

epochs = 1
change_level = 3


#分類に使う配列
class_arr1 = np.array([[1,1,1],[1,0,1],[1,1,1]])
class_arr2 = np.array([[0,1,0],[1,0,1],[0,1,0]])
class_arr3 = np.array([[1,1,1],[0,0,0],[0,0,0]])
class_arr4 = np.array([[0,0,0],[0,0,0],[1,1,1]])
class_arr5 = np.array([[0,0,0],[1,0,1],[0,0,0]])

#初期個体生成
def create_first_design():
    design = np.ones((y_len+2,x_len+2))
    design[1:y_len+1, 1:x_len+1] = 0
    #最初の形を決める
    #design[:, 13] = 1
    design[1:7,5:18] = 2
    design[1:7,4] = 1
    design[1:7,18] = 1
    design[7,5:18] = 1
    return design

#近傍解生成
def create_neighbors(design):
    pattern = 0
    for cl in range (change_level):
        #変更箇所選出
        if pattern > 0:
            next = np.array(list(zip(*np.where(change_area==1))))
            loop = True
            for n in range(next.shape[0]):
                cp_x = cp_x -1 + next[n][1]
                cp_y = cp_y -1 + next[n][0]
                if 0<cp_x<x_len+1 and 0<cp_y<x_len+1:
                    loop = False
                    break
            if loop:
                pattern = 0
        if pattern == 0:
            #境界部抽出
            groove = np.array(list(zip(*np.where(design[1:y_len+1,1:x_len+1]==1))))+1
            #変更箇所選定
            n = random.randrange(0,len(groove))
            cp_x = groove[n][1]
            cp_y = groove[n][0]
        
        
        
        change_area = design[cp_y-1:cp_y+2, cp_x-1:cp_x+2]

        #変更
        print(cp_x,cp_y,cl)
        print(change_area)
        _change_area = np.where(change_area>1, 0, change_area)
        print(_change_area)
        pattern = np.sum(class_arr1*_change_area)
        print("pattern",pattern)
        subpattern = np.sum(class_arr2*_change_area)
        print("subpattern", subpattern)
        #分岐
        if pattern == 0:
            change_area[1][1] = change_area[0][0]
        
        elif pattern == 1:
            if _change_area[0][0] == 0:
                change_area[1][1] = change_area[0][0]
            elif _change_area[0][1] == 0:
                change_area[1][1] = change_area[0][1]
        
        elif pattern == 2:
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
                elif np.sum(class_arr3*_change_area) == 1:
                    change_area[1][1] = change_area[2][1]
                elif np.sum(class_arr4*_change_area) == 1:
                    change_area[1][1] = change_area[0][1]
            
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
            
        elif pattern == 3:
            if subpattern == 0:#G　未実装
                print("3-G")
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
                        print("3-H")
                elif np.sum(class_arr4*_change_area) >= 2:
                    change_area[1][1] = change_area[0][1]
                    if _change_area[0][0] == 1:
                        change_area[1][0] = 1
                    elif _change_area[0][2] == 1:
                        change_area[1][2] = 1
                    elif np.sum(class_arr5*_change_area) == 1:
                        change_area[2][1] = 1
                    elif _change_area[0][1] == 1:
                        print("3-H")
                elif np.sum(class_arr3.T*_change_area) >= 2:
                    change_area[1][1] = change_area[1][2]
                    if _change_area[0][2] == 1:
                        change_area[0][1] = 1
                    elif _change_area[2][2] == 1:
                        change_area[2][1] = 1
                    elif _change_area[1][2] == 1:
                        print("3-H")
                elif np.sum(class_arr4.T*_change_area) >= 2:
                    change_area[1][1] = change_area[1][0]
                    if _change_area[0][0] == 1:
                        change_area[0][1] = 1
                    elif _change_area[2][0] == 1:
                        change_area[2][1] = 1
                    elif _change_area[1][0] == 1:
                        print("3-H")
            
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
        #この先未実装
        elif pattern == 100:#4
            #_change_area[1][1] = 0
            if subpattern == 0:#L
                print("4-L")
            
            elif subpattern == 1:#BJ
                #B
                _change_area[1][1] = 0
                if np.sum(class_arr3*_change_area) == 3:
                    if _change_area[2][0] == 1:
                        _change_area[1][0] = 1
                    elif _change_area[2][2] == 1:
                        _change_area[1][2] = 1
                elif np.sum(class_arr4*_change_area) == 3:
                    if _change_area[0][0] == 1:
                        _change_area[1][0] = 1
                    elif _change_area[0][2] == 1:
                        _change_area[1][2] = 1
                elif np.sum(class_arr3.T*_change_area) == 3:
                    if _change_area[0][2] == 1:
                        _change_area[0][1] = 1
                    elif _change_area[2][2] == 1:
                        _change_area[2][1] = 1
                elif np.sum(class_arr4.T*_change_area) == 3:
                    if _change_area[0][0] == 1:
                        _change_area[0][1] = 1
                    elif _change_area[2][0] == 1:
                        _change_area[2][1] = 1
                #J
                else:
                    print("4-J")
                    _change_area[1][1] = 1

            elif subpattern == 2:#ACEFHIK
                #IK
                if np.sum(class_arr5*_change_area) == 2:
                    if np.sum(class_arr3*_change_area) == 2:
                        _change_area[0][1] = 1
                    elif np.sum(class_arr4*_change_area) == 2:
                        _change_area[2][1] = 1
                    else:
                        dise = random.randint(0,1)
                        if dise == 0:
                            _change_area[0][1] = 1
                        elif dise == 1:
                            _change_area[2][1] = 1
                elif np.sum(class_arr5.T*_change_area) == 2:
                    if np.sum(class_arr3.T*_change_area) == 2:
                        _change_area[1][2] = 1
                    elif np.sum(class_arr4.T*_change_area) == 2:
                        _change_area[1][0] = 1
                    else:
                        dise = random.randint(0,1)
                        if dise == 0:
                            _change_area[1][0] = 1
                        elif dise == 1:
                            _change_area[1][2] = 1
                #A
                
            #elif subpattern == 3:#DG
        
        else:
            print(pattern,"が出ました")
            print(_change_area)

        print(change_area,cl)
    neighbors = design
    return neighbors


def main():
    #初期個体生成
    design = create_first_design()
    #評価
    #model = load_model(model_path)
    #eval = model.predict(design)

    for i in range(epochs):
        #近傍解生成
        #評価
        #入れ替え
        print(i)
    
    print(design)


if __name__ == '__main__':
    map = design.main()
    #map = create_first_design()
    #print(map)

    neighbors = create_neighbors(map)
    print(map)
    