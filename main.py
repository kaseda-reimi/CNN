import numpy as np
from tensorflow.keras.models import load_model

from nn import model_path
#from cnn import model_path

x_size = 28
y_size = 28

epochs = 1

goal = 9

#初期個体生成
def create_first_design():
    design = np.zeros((y_size,x_size))
    design[:, 13] = 1
    return design

#近傍解生成
def create_neighbors(design):
    neighbors = design
    
    return neighbors

def main():
    #初期個体生成
    design = create_first_design()
    #評価
    model = load_model(model_path)
    eval = model.predict(design)

    for i in range(epochs):
        #近傍解生成
        #評価
        #入れ替え
    
    print(design)


if __name__ == '__main__':
    main()