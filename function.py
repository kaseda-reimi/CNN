import os
import numpy as np
import random
#import matplotlib.pyplot as plt

x_len = 40
y_len = 6

def get_data():
    path = os.getcwd()+'/data_r2.txt'
    with open (path) as f:
        l = f.read().split()
    path = os.getcwd()+'/data.txt'
    with open (path) as f:
        l.extend(f.read().split())
    
    data = [float(s) for s in l]
    data = np.array(data).reshape(-1, x_len*y_len+2)
    structure = data[:,:x_len*y_len].reshape(-1, y_len, x_len)
    output = data[:,x_len*y_len:].reshape(-1, 2)
    half_str, half_out = get_data_half()
    structure = np.concatenate([structure, half_str])
    output = np.concatenate([output, half_out])
    #output_ratio = np.zeros([output.shape[0],2])
    #output_ratio[:,0] = 2 * np.log10(output[:,0]/output[:,1]) #消光比/10
    #output_ratio[:,1] = 2 * np.log10(1/output[:,0]) #挿入損失/10
    output[:,1] = output[:,1]/output[:,0]
    #上下反転してデータを増やす
    #structure = np.concatenate([structure, np.flip(structure,1)])
    #output_ratio = np.concatenate([output_ratio, output_ratio])
    #output = np.concatenate([output, output])
    return structure, output#_ratio

def get_data_half():
    half_x = 20
    path = os.getcwd()+'/data_half.txt'
    with open (path) as f:
        l = f.read().split()
    data = [float(s) for s in l]
    data = np.array(data).reshape(-1, half_x*y_len+2)
    structure = data[:,:half_x*y_len].reshape(-1, y_len, half_x)
    output = data[:,half_x*y_len:].reshape(-1, 2)
    input = np.zeros([structure.shape[0], y_len, x_len])
    for n in range(structure.shape[0]):
        for j in range(y_len):
            for i in range(half_x):
                input[n][j][i*2:i*2+2] = structure[n][j][i]
    for n in range(input.shape[0]):
        for j in range(y_len):
            for i in range(x_len):
                if input[n,j,i] == 1:
                    up = j-1
                    down = j+2
                    left = i-1
                    right = i+2
                    if up < 0:
                        up = 0
                    if down > y_len:
                        down = y_len
                    if left < 0:
                        left = 0
                    if right > x_len:
                        right = x_len
                    if np.prod(input[n,up:down,left:right]-2) != 0:
                        input[n][j][i] = 0
                    
    output_ratio = np.zeros([output.shape[0],2])
    output_ratio[:,0] = 2 * np.log10(output[:,0]/output[:,1]) #消光比/10
    output_ratio[:,1] = 2 * np.log10(1/output[:,0]) #挿入損失/10
    return input, output#, output_ratio

def normalize(x):
    normalized_x= (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    return normalized_x, np.amin(x), np.amax(x)

def evaluation(y):
    #非正規化
    #y[0,0] = y[0,0]*(0.65-0.06) + 0.06
    #y[0,1] = y[0,1]*(0.65-0.01) + 0.01
    if y[0,0] < 0:
        y[0,0] = 0.01
    if y[0,1] < 0:
        y[0,1] = 0.01
    #消光比計算
    extinction = 20 * np.log10(1/y[0,1])
    loss = 20 * np.log10(1/y[0,0])
    E = extinction# - loss
    return E, extinction, loss

def write_data(path, data):
    with open(os.getcwd()+path, mode='w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for n in data[i][j]:
                    f.write(str(n)+" ")
                f.write('\n')
            f.write('\n')
    print("了")

def search_E_max(data):
    E_max = -10
    max_index = 0
    for i in range(data.shape[0]):
        E = evaluation(data[i])
        if E > E_max:
            E_max = E
            max_index = i
    return E_max, max_index

def design():
    EA_min = x_len * y_len * 0.05
    EA_max = x_len * y_len * 0.4
    EA = np.random.randint(EA_min, EA_max)
    design = np.zeros((y_len,x_len))
    #初期位置
    x = np.random.randint(0, x_len)
    y = 0
    #形成
    for i in range(1,EA):
        design[y][x] = 2
        #周囲を1に
        if 0<y<y_len :
            if design[y-1][x]==0 :
                design[y-1][x] = 1
        if y<y_len-1 :
            if design[y+1][x]==0 :
                design[y+1][x] = 1
        if 0<x<x_len :
            if design[y][x-1]==0 :
                design[y][x-1] = 1
        if x<x_len-1 :
            if design[y][x+1]==0 :
                design[y][x+1] = 1
        order = np.random.rand(4)
        #1：上, 2：右, 3：下, 4：左
        direction_x = [x, x+1, x, x-1]
        direction_y = [y-1, y, y+1, y]
        count = 0
        while count < 5:
            direction = np.argmax(order)
            _x = direction_x[direction]
            _y = direction_y[direction]

            if 0<=_x<x_len and 0<=_y<y_len/2 :
                if design[_y][_x] == 1 :
                    y, x = _y, _x
                    break
            #else
            order[direction] = 0
            count = count + 1
            if count == 4:
                for i in range(0,x_len):
                    if design[y][i] == 1:
                        x = i
                        break
                if design[y][x] != 1:
                    for j in range(0, y_len/2):
                        if design[j][x] == 1:
                            y = j
                            break
                #保険
                if design[y][x] != 1:
                    flag = False
                    print("例外")
                    for i in range(0, x_len):
                        for j in range(0, y_len/2):
                            if design[j][i] == 1:
                                x, y = i, j
                                flag = True
                                break
                        if flag:
                            break
                count = 5
    #１で囲む
    print(EA)
    design = np.insert(design, y_len, 1, axis=0)
    design = np.insert(design, 0, 1, axis=0)
    design = np.insert(design, x_len, 1, axis=1)
    design = np.insert(design, 0, 1, axis=1)
    #穴埋め
    groove = np.array(list(zip(*np.where(design[1:y_len+1,1:x_len+1]==1))))+1
    num = 0
    for n in groove:
        area = design[n[0]-1:n[0]+2, n[1]-1:n[1]+2]
        if np.prod(area) > 0:
            area[1][1] = 2
            num += 1
    return design[1:y_len+1, 1:x_len+1]

def create_neighbor(design):
    for change_level in range(3):
        #境界部抽出
        groove = np.array(list(zip(*np.where(design[:,:]==1))))+1
        #変更箇所選定
        n = random.randrange(0,len(groove))
        x = groove[n][1]
        y = groove[n][0]
        dise = random.randint(0,1)
        if dise == 0:
            design[y][x] = 0
            if y < y_len-1 and design[y+1][x] == 2:
                design[y+1][x] = 1
            if y > 0 and design[y-1][x] == 2:
                design[y-1][x] = 1
            if x < x_len-1 and design[y][x+1] == 2:
                design[y][x+1] = 1
            if x > 0 and design[y][x-1] == 2:
                design[y][x-1] = 1
        if dise == 1:
            if y < y_len-1:
                design[y][x] = 2
                if design[y+1][x] == 0:
                    design[y+1][x] = 1
                if y > 0 and design[y-1][x] == 0:
                    design[y-1][x] = 1
                if x > 0 and design[y][x-1] == 0:
                    design[y][x-1] = 1
                if x < x_len-1 and design[y][x+1] == 0:
                    design[y][x+1] = 1
        groove = np.array(list(zip(*np.where(design[:,:]==1))))+1
        for n in groove:
            if design[n[0]-1][n[1]]!=2 and design[n[0]+1][n[1]]!=2 and design[n[0]][n[1]-1]!=2 and design[n[0]][n[1]+1]!=2:
                design[n[0]][n[1]] = 0
    return design


if __name__ == '__main__':
    design = design()
    print(design)
    create_neighbor(design)
    print(design)