import numpy as np
import random
import os
import copy
from tensorflow.keras.models import load_model
from nn import model_path
#from cnn import model_path
import function as fc
from function import x_len, y_len


epochs = 50
group = 20
change_level = 3

#初期個体生成
def create_first_design(mode):
    design = np.zeros((y_len,x_len))
    #最初の形を決める
    if mode == 0:
        design = fc.design()
    if mode == 1:
        #design[:, 13] = 1
        design[0:7,5:18] = 2
        design[0:7,4] = 1
        design[0:7,18] = 1
        design[0,5:18] = 1
    return design

def main():
    #初期個体生成
    design = fc.get_design()
    #評価
    model = load_model(model_path)
    perform = model.predict(design.reshape(1, -1))
    eval = fc.evaluation(perform[0])
    start = copy.copy(design)
    eval_start = copy.copy(eval)
    for i in range(epochs):
        best_design = design
        best_eval = eval
        for _ in range(group):
            neighbor = fc.create_neighbor(design)
            nei_perform = model.predict(neighbor.reshape(1,-1))
            nei_eval = fc.evaluation(nei_perform)
            if nei_eval[0] > best_eval[0]:
                best_eval = nei_eval
                best_design = neighbor
        design = best_design
        eval = best_eval
        print(i, best_eval)
        #if best_eval[0] > 40:
        #    break
    perform = model.predict(design.reshape(1,-1))
    eval = fc.evaluation(perform)
    for j in range(i+1,epochs):
        best_design = design
        best_eval = eval
        for _ in range(group):
            neighbor = fc.create_neighbor(design)
            nei_perform = model.predict(neighbor.reshape(1,-1))
            nei_eval = fc.evaluation(nei_perform)
            if nei_eval[0] > best_eval[0]:
                best_eval = nei_eval
                best_design = neighbor
        design = best_design
        eval = best_eval
        print(j, best_eval)
    
    print(start)
    print(eval_start)
    print(design)
    print(best_eval)
    print(model.predict(design.reshape(1,-1)))


if __name__ == '__main__':
    main()