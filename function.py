import os
import numpy as np
from design import x_len, y_len

def get_data():
    path = os.getcwd()+'/data_r.txt'
    with open (path) as f:
        l = f.read().split()
    data = [float(s) for s in l]
    data = np.array(data).reshape(-1, x_len*y_len+2)
    input_data = data[:,:x_len*y_len]
    output_data = data[:,x_len*y_len:]
    return input_data, output_data

if __name__ == '__main__':
    x, y = get_data()
    print(x[0])
    print(y[0])