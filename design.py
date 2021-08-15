#for Lumerical

import numpy as np

x_len = 40
y_len = 6
EA_min = x_len * y_len * 0.1
EA_max = x_len * y_len * 0.4


def main():
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

            if 0<=_x<x_len and 0<=_y<y_len :
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
                    for j in range(0, y_len):
                        if design[j][x] == 1:
                            y = j
                            break
                #保険
                if design[y][x] != 1:
                    flag = False
                    print("例外")
                    for i in range(0, x_len):
                        for j in range(0, y_len):
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
    
    print(design)
    #穴埋め
    groove = np.array(list(zip(*np.where(design[1:y_len+1,1:x_len+1]==1))))+1
    num = 0
    for n in groove:
        area = design[n[0]-1:n[0]+2, n[1]-1:n[1]+2]
        if np.prod(area) > 0:
            area[1][1] = 2
            num += 1
    print(num)
    print(design)
    return design

if __name__ == '__main__':
    main()
    