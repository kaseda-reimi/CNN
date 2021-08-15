import os
import numpy as np
from design import x_len, y_len

def get_data():
    path = os.getcwd()+'/data_r.txt'
    with open (path) as f:
        l = f.read().split()
    data = [float(s) for s in l]
    data = np.array(data).reshape(-1, x_len*y_len+2)
    structure = data[:,:x_len*y_len].reshape(-1, x_len*y_len)
    output = data[:,x_len*y_len:].reshape(-1, 2)
    output_ratio = np.zeros([output.shape[0],2])
    output_ratio[:,0] = 2 * np.log10(output[:,0]/output[:,1]) #消光比/10
    output_ratio[:,1] = 2 * np.log10(1/output[:,0]) #挿入損失/10
    return structure, output_ratio

if __name__ == '__main__':
    x, y = get_data()
    print(x[0])
    print(y[0])
    