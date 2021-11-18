
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import function as fc
import numpy as np
from design import x_len, y_len #40,6

input_size = y_len*x_len
epochs = 1000
batch_size = 128

model_path = os.getcwd()+'/nn_model'

def main():
    input_data, output_data = fc.get_data()
    input_data  = input_data.astype('float32')
    input_data = input_data.reshape(-1, input_size)
    #正規化
    input_data /= 2
    output_data[:,0] = fc.normalize(output_data[:,0])
    output_data[:,1] = fc.normalize(output_data[:,1])

    x_train, x_test, y_train, y_test = train_test_split(input_data,output_data,test_size=0.1)

    model = Sequential()
    model.add(InputLayer(input_shape=(input_size,)))#240
    model.add(Dense(input_size/2, activation = "relu"))#120
    model.add(Dense(input_size/2, activation = "relu"))#120
    model.add(Dense(input_size/2, activation = "relu"))#120
    model.add(Dense(input_size/3, activation = "relu"))#80
    model.add(Dense(2, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test)
    )

    score = model.evaluate(x_test, y_test, verbose=1)
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    nb_epoch = len(loss)
    plt.plot(range(nb_epoch), loss,     marker='.', label='loss')
    plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("nn_learning.png")

    model.save(model_path)

    predict = model.predict(x_test)
    for i in range(y_test.shape[0]):
        print(y_test[i], predict[i])
        
    print(np.corrcoef(y_test[:,0].reshape(1,-1), predict[:,0].reshape(1,-1)))
    print(np.corrcoef(y_test[:,1], predict[:,1]))


if __name__ == '__main__':
    main()