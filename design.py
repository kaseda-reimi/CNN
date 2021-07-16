#for Lumerical

import numpy as np

x_max = 20
y_max = 30
EA_min = x_max * y_max * 0.1
EA_max = x_max * y_max * 0.5

def main():
    EA = np.random.randint(EA_min, EA_max)
    design = np.zeros((y_max,x_max))
    #初期位置
    x = np.random.randint(0, x_max)
    y = 0

    for i in range(1,EA):
        design[y][x] = 2
        if 0<y<y_max :
            if design[y-1][x]==0 :
                design[y-1][x] = 1
        if y<y_max-1 :
            if design[y+1][x]==0 :
                design[y+1][x] = 1
        if 0<x<x_max :
            if design[y][x-1]==0 :
                design[y][x-1] = 1
        if x<x_max-1 :
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

            if 0<=_x<x_max and 0<=_y<y_max :
                if design[_y][_x] == 1 :
                    y, x = _y, _x
                    break
            #else
            order[direction] = 0
            count = count + 1
            if count == 4:
                #pan, pasta, udon, small pasuta, syanpu-
                for i in range(0,x_max):
                    if design[y][i] == 1:
                        x = i
                        break
                if design[y][x] != 1:
                    for j in range(0, y_max):
                        if design[j][x] == 1:
                            y = j
                            break
                #保険
                if design[y][x] != 1:
                    flag = False
                    print("例外")
                    for i in range(0, x_max):
                        for j in range(0, y_max):
                            if design[j][i] == 1:
                                x, y = i, j
                                flag = True
                                break
                        if flag:
                            break

                count = 5
                
            
        
    print(design)
    print(EA)

if __name__ == '__main__':
    main()