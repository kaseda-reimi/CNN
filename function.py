import os
import numpy as np
from design import x_len, y_len

def get_data():
    path = os.getcwd()+'/data_r.txt'
    with open (path) as f:
        l = f.read().split()
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

def normalize(x):
    normalized_x= (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    return normalized_x

def evaluation(y):
    E = y[0] / y[1]
    return E

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
    x, y = get_data()
    print(y.shape[0])
    E_max, max_index = search_E_max(y)
    print(E_max)
    print(x[max_index])
    print(y[max_index])
    print(max_index)
    #y[:,0] = normalize(y[:,0])
    #print(y[:,0])

    