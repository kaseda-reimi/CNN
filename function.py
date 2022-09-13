import os
import numpy as np
import random
import copy
#import matplotlib.pyplot as plt

x_len = 39
y_len = 6

def get_data_old():
    path = os.getcwd()+'/data.txt'
    with open (path) as f:
        l = f.read().split()
    path = os.getcwd()+'/data_r2.txt'
    with open (path) as f:
        l.extend(f.read().split())
    
    data = [float(s) for s in l]
    data = np.array(data).reshape(-1, 40*y_len+2)
    structure = data[:,:40*y_len].reshape(-1, y_len, 40)
    output = data[:,40*y_len:].reshape(-1, 2)
    half_str, half_out = get_data_half()
    structure = np.concatenate([structure, half_str])
    output = np.concatenate([output, half_out])
    #output_ratio = np.zeros([output.shape[0],2])
    #output_ratio[:,0] = 2 * np.log10(output[:,0]/output[:,1]) #消光比/10
    #output_ratio[:,1] = 2 * np.log10(1/output[:,0]) #挿入損失/10
    output[:,1] = output[:,1]/output[:,0]
    
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
    input = np.zeros([structure.shape[0], y_len, 40])
    for n in range(structure.shape[0]):
        for j in range(y_len):
            for i in range(half_x):
                input[n][j][i*2:i*2+2] = structure[n][j][i]
    for n in range(input.shape[0]):
        for j in range(y_len):
            for i in range(40):
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
                    if right > 40:
                        right = 40
                    if np.prod(input[n,up:down,left:right]-2) != 0:
                        input[n][j][i] = 0
                    
    output_ratio = np.zeros([output.shape[0],2])
    output_ratio[:,0] = 2 * np.log10(output[:,0]/output[:,1]) #消光比/10
    output_ratio[:,1] = 2 * np.log10(1/output[:,0]) #挿入損失/10
    return input, output#, output_ratio

def get_data():
    path = os.getcwd()+'/data_78x.txt'
    with open (path) as f:
        l = f.read().split()
    path = os.getcwd()+'/data_78.txt'
    with open (path) as f:
        l.extend(f.read().split())
    data = [float(s) for s in l]
    data = np.array(data).reshape(-1, x_len*y_len+2)
    structure = data[:,:x_len*y_len].reshape(-1, y_len, x_len)
    output = data[:,x_len*y_len:].reshape(-1, 2)
    output[:,1] = output[:,1]/output[:,0]

    return structure, output

def normalize(x):
    normalized_x= (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    return normalized_x, np.amin(x), np.amax(x)

def evaluation(y):
    if y[0] < 0:
        y[0] = 0.001
    if y[1] < 0:
        y[1] = 0.001
    extinction = 20 * np.log10(1/y[1])
    loss = 20 * np.log10(1/y[0])
    a = 1
    b = 2
    _ex = extinction
    if (extinction > 20):
        _ex = 20
    E = a * _ex - b * loss
    return E, extinction, loss

def evaluation_2(x,y):
    if y[0] < 0:
        y[0] = 0.001
    if y[1] < 0:
        y[1] = 0.001
    extinction = 20 * np.log10(1/y[1])
    loss = 20 * np.log10(1/y[0])
    #groove = np.count_nonzero(x==1)
    groove = count_groove(x)
    a = 1
    b = 0
    c = 0.3
    E = a * extinction - b * loss - c * groove
    return E, extinction, loss, groove

def count_groove(design):
    groove = 0
    for j in range(y_len):
        for i in range(x_len):
            if design[j,i] == 1:
                if j+1 < y_len:
                    if design[j+1][i] == 2:
                        groove += 2
                if j-1 >= 0:
                    if design[j-1][i] == 2:
                        groove += 2
                if i-1 >= 0:
                    if design[j][i-1] == 2:
                        groove += 1
                if i+1 < x_len:
                    if design[j][i+1] == 2:
                        groove +=1
    return groove


def write_data(path, data, output):
    with open(os.getcwd()+"/"+path+".txt", mode='w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for n in data[i][j]:
                    f.write(str(int(n))+" ")
                f.write('\n')
            #f.write(str(output[i][0])+"\n"+str(output[i][1]))
            f.write('\n')
    print("了")

def search_E_max(input, output):
    E_max = -10
    max_index = 0
    for i in range(input.shape[0]):
        E = evaluation_2(input[i],output[i])
        if E[0] > E_max:
            E_max = E[0]
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
    #print(EA)
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

def get_design():
    input, output = get_data()
    #n = np.argmin(output[:,1])
    #n = 21,520
    n = 394
    design = input[n]
    return design

def get_design_z():
    design = np.zeros((y_len, x_len))
    design[0] = [2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    design[1] = [2,2,2,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
    design[2] = [2,2,1,0,0,0,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,2,2,2,2,2,2,2,2,2,2]
    design[3] = [1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
    design[4] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    design[5] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return design

def create_neighbor(design,change_level):
    neighbor = copy.copy(design)
    for _ in range(change_level):
        #境界部抽出
        groove = np.array(list(zip(*np.where(neighbor[:,:]==1))))
        #変更箇所選定
        n = random.randrange(0,len(groove))
        x = groove[n][1]
        y = groove[n][0]
        #print(n,y,x)
        dise = random.randint(0,1)
        if y > y_len/2:
            dise = 0
        if dise == 0:
            neighbor[y][x] = 0
            if y < y_len-1 and neighbor[y+1][x] == 2:
                neighbor[y+1][x] = 1
            if y > 0 and neighbor[y-1][x] == 2:
                neighbor[y-1][x] = 1
            if x < x_len-1 and neighbor[y][x+1] == 2:
                neighbor[y][x+1] = 1
            if x > 0 and neighbor[y][x-1] == 2:
                neighbor[y][x-1] = 1
        if dise == 1:
            neighbor[y][x] = 2
            if neighbor[y+1][x] == 0:
                neighbor[y+1][x] = 1
            if y > 0 and neighbor[y-1][x] == 0:
                neighbor[y-1][x] = 1
            if x > 0 and neighbor[y][x-1] == 0:
                neighbor[y][x-1] = 1
            if x < x_len-1 and neighbor[y][x+1] == 0:
                neighbor[y][x+1] = 1
        groove = np.array(list(zip(*np.where(neighbor[:,:]==1))))
        for g in groove:
            neighbor[g[0]][g[1]] = 0
            if g[0]>0 and neighbor[g[0]-1][g[1]]==2:
                neighbor[g[0]][g[1]] = 1
            elif g[0]<y_len-1 and neighbor[g[0]+1][g[1]]==2:
                neighbor[g[0]][g[1]] = 1
            elif g[1]>0 and neighbor[g[0]][g[1]-1]==2:
                neighbor[g[0]][g[1]] = 1
            elif g[1]<x_len-1 and neighbor[g[0]][g[1]+1]==2:
                neighbor[g[0]][g[1]] = 1
            
    return neighbor

def distribution(data):
    #data = 20*np.log10(1/data)
    data = 10 * data
    print(np.amax(data))
    print(np.amin(data))
    distribution = np.zeros([2,10])
    for i in range(data.shape[0]):
        n = int(data[i][0] // 1)
        m = int(data[i][1] // 1)
        distribution[0][n] += 1
        distribution[1][m] += 1
    return distribution

if __name__ == '__main__':
    input, output = get_data()
    E, n = search_E_max(input, output)
    print(E, n)
    #print(input[21])
    #print(input.shape[0])
    print(input[n])
    print(20*np.log10(1/output[n]))
