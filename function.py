import os
import numpy as np
#import matplotlib.pyplot as plt
from design import x_len, y_len

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
    #上下反転してデータを増やす
    structure = np.concatenate([structure, np.flip(structure,1)])
    #output_ratio = np.concatenate([output_ratio, output_ratio])
    output = np.concatenate([output, output])
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
    E = y[0] - y[1]
    return E

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





if __name__ == '__main__':
    input, output1= get_data()
    #print(np.amin(output1))
    #print(np.amax(output1))
    #x = list(range(0, 10, 10))
    #y = np.zeros([2,10])
    #for i in range(output1.shape[0]):
    #    a = output1[i][0]//0.1
    #    y[0][int(a)] += 1
    #    b = output1[i][1]//0.1
    #    y[1][int(b)] += 1
    #print(y)
    #plt.plot(x, y[0], marker="o", color = "red", linestyle = "--")
    #plt.plot(x, y[1], marker="o", color = "blue", linestyle = "--")
    #plt.savefig("data_hikaku.png")
    #print(input[0])
    print(input.shape[0])