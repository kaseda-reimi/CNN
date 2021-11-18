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
    output_ratio = np.zeros([output.shape[0],2])
    output_ratio[:,0] = 2 * np.log10(output[:,0]/output[:,1]) #消光比/10
    output_ratio[:,1] = 2 * np.log10(1/output[:,0]) #挿入損失/10
    #上下反転してデータを増やす
    structure = np.concatenate([structure, np.flip(structure,1)])
    output_ratio = np.concatenate([output_ratio, output_ratio])
    return structure, output_ratio

def get_data_half():
    x_len = 20
    path = os.getcwd()+'/data_half.txt'
    with open (path) as f:
        l = f.read().split()
    data = [float(s) for s in l]
    data = np.array(data).reshape(-1, x_len*y_len+2)
    structure = data[:,:x_len*y_len].reshape(-1, y_len, x_len)
    output = data[:,x_len*y_len:].reshape(-1, 2)
    #output_ratio = output**2
    output_ratio = np.zeros([output.shape[0],2])
    output_ratio[:,0] = 2 * np.log10(output[:,0]/output[:,1]) #消光比/10
    output_ratio[:,1] = 2 * np.log10(1/output[:,0]) #挿入損失/10
    return structure, output, output_ratio

def normalize(x):
    normalized_x= (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    return normalized_x

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
    input, output1 = get_data()
    print(np.amin(output1))
    print(np.amax(output1))
    #print(input.shape)
    #print(output1[1])
    x = list(range(0, 10, 10))
    y = np.zeros([2,10])
    for i in range(output1.shape[0]):
        a = output1[i][0]//0.1
        y[0][int(a)] += 1
        b = output1[i][1]//0.1
        y[1][int(b)] += 1
    print(y)

    #plt.plot(x, y[0], marker="o", color = "red", linestyle = "--")
    #plt.plot(x, y[1], marker="o", color = "blue", linestyle = "--")
    #plt.savefig("data_hikaku.png")