
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import function as fc
from function import y_len, x_len
import numpy as np

input_size = y_len*x_len
epochs = 2000
batch_size = 64

model_path = os.getcwd()+'/nn_model'

def main():
    input_data, output_data = fc.get_data()
    input_data  = input_data.astype('float32')
    input_data = input_data.reshape(-1, input_size)
    #正規化
    input_data /= 2
    #output_data[:,0], min0, max0 = fc.normalize(output_data[:,0])
    #output_data[:,1], min1, max1 = fc.normalize(output_data[:,1])

    x_train, x_test, y_train, y_test = train_test_split(input_data,output_data,test_size=0.1)

    model = Sequential()
    model.add(InputLayer(input_shape=(input_size,)))
    model.add(Dense(input_size/1, activation = "relu"))
    model.add(Dense(input_size/1, activation = "relu"))
    model.add(Dense(input_size/1, activation = "relu"))
    #model.add(Dense(input_size/2, activation = "relu"))
    #model.add(Dense(input_size/3, activation = "relu"))
    #model.add(Dense(input_size/6, activation = "relu"))
    model.add(Dense(2, activation='sigmoid'))
    
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
    plt.yscale("log")
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("nn_learning.png")

    model.save(model_path)

    predict = model.predict(x_test)
    for i in range(y_test.shape[0]):
        print(y_test[i], predict[i])
    
        
    print(np.corrcoef(y_test[:,0], predict[:,0]))
    print(np.corrcoef(y_test[:,1], predict[:,1]))
    #非正規化
    #y_test[:,0] = y_test[:,0] * (max0 - min0) + min0
    #predict[:,0] = predict[:,0] * (max0 - min0) + min0
    #y_test[:,1] = y_test[:,1] * (max1 - min1) + min1
    #predict[:,1] = predict[:,1] * (max1 - min1) + min1

    model.summary()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    ax.scatter(y_test[:,0],predict[:,0], c='red')
    ax.scatter(y_test[:,1],predict[:,1], c='blue')
    ax.set_xlabel('simulation')
    ax.set_ylabel('NN')
    plt.savefig("scatter_.png")

    simulation = np.zeros(y_test.shape)
    simulation[:,0] = 2 * np.log10(1/y_test[:,0])
    simulation[:,1] = 2 * np.log10(1/y_test[:,1])
    nn = np.zeros(predict.shape)
    nn[:,0] = 2 * np.log10(1/predict[:,0])
    nn[:,1] = 2 * np.log10(1/predict[:,1])

    print(np.corrcoef(simulation[:,0], nn[:,0]))
    print(np.corrcoef(simulation[:,1], nn[:,1]))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    ax.scatter(simulation[:,0],nn[:,0], c='red')
    ax.scatter(simulation[:,1],nn[:,1], c='blue')
    ax.set_xlabel('simulation')
    ax.set_ylabel('NN')
    plt.savefig("scatter.png")


if __name__ == '__main__':
    main()