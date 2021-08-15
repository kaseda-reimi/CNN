
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import function as fc
from design import x_len, y_len

epochs = 20
batch_size = 128

model_path = os.getcwd()+'/nn_model'

def main():
    input_data, output_data = fc.get_data()
    input_data  = input_data.reshape(-1, x_len*y_len)
    output_data = output_data.rexhape(-1, 2)
    input_data  = input_data.astype('float32')
    input_data /= 2
    x_train, x_test, y_train, y_test = train_test_split(input_data,output_data,test_size=0.1)

    model = Sequential()
    model.add(InputLayer(input_shape=(x_len*y_len,)))
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

if __name__ == '__main__':
    main()